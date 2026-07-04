"""
3Blue1Brown-style Manim animation explaining iterative imputation for
demand unconstraining of out-of-stock (OOS) hours.

Story:
  - We have hourly sales. Some hours are OOS -> observed sales are censored,
    so the *true demand* is hidden and must be "unconstrained" (imputed).
  - Step 1 (initialize): fill each OOS hour with the mean hourly sales of the
    last 7 days.
  - Step 2 (iterate, Gauss-Seidel style): with two missing hours x (hour 13)
    and y (hour 14), use x + all other hours to re-predict y -> y'; then use
    y' + all other hours to re-predict x -> x'; repeat for 3 rounds until the
    estimates stabilise.

Scenes (render each separately so you can drop them onto individual slides):
  manim -pqh demand_unconstraining.py TableIntro
  manim -pqh demand_unconstraining.py InitialImpute
  manim -pqh demand_unconstraining.py IterativeImpute
  manim -pqh demand_unconstraining.py Convergence

Or render everything continuously:
  manim -pqh demand_unconstraining.py FullStory

Tested with Manim Community v0.18+.
"""

from manim import *

# ----------------------------------------------------------------------------
# Palette (3b1b-ish on a black background)
# ----------------------------------------------------------------------------
HEADER_C = BLUE_B          # column / row headers
KNOWN_C = WHITE            # observed sales
OOS_C = "#E74C3C"          # out-of-stock (censored) cells
MEAN_C = "#F1C40F"         # the 7-day mean initialisation
PRED_C = "#2ECC71"         # freshly predicted / updated value
FEATURE_C = "#3AAFA9"      # cells used as features for a prediction
MODEL_C = "#BB86FC"        # the regression model f(.)

# ----------------------------------------------------------------------------
# Problem data (illustrative numbers chosen to show a clean convergence)
# ----------------------------------------------------------------------------
HOURS = [12, 13, 14, 15, 16, 17]
SALES = ["5", "OOS", "OOS", "4", "3", "1"]
OOS_IDX = [1, 2]                 # hour 13 (x) and hour 14 (y)
X_COL, Y_COL = 1, 2              # column indices of x and y

# 7-day means used for initialisation
X_MEAN, Y_MEAN = 8.0, 7.0

# Iterative-imputation trajectory (x = hour 13, y = hour 14)
#   round 0 = initialisation, then 3 refinement rounds
X_SERIES = [8.0, 7.4, 7.1, 7.0]
Y_SERIES = [7.0, 6.4, 6.7, 6.8]


def fit(mobj, max_w):
    """Scale a mobject down so it fits inside max_w."""
    if mobj.width > max_w:
        mobj.scale(max_w / mobj.width)
    return mobj


def fmt(v):
    return f"{v:.1f}"


class ImputationBase(Scene):
    """Shared helpers: builds the hourly-sales table and exposes references."""

    label_w = 3.0
    cell_w = 1.15
    cell_h = 1.0

    def make_cell(self, w):
        return Rectangle(
            width=w, height=self.cell_h,
            stroke_color=GREY_B, stroke_width=2.5,
            fill_color=BLACK, fill_opacity=1.0,
        )

    def build_table(self, table_shift=ORIGIN):
        n = len(HOURS)
        total_w = self.label_w + n * self.cell_w
        left = -total_w / 2.0
        sales_y = self.cell_h / 2.0
        hour_y = -self.cell_h / 2.0

        def data_x(j):
            return left + self.label_w + (j + 0.5) * self.cell_w

        label_x = left + self.label_w / 2.0

        self.table = VGroup()
        self.sales_rects = []
        self.sales_labels = []   # value mobjects (editable)
        self.hour_rects = []
        self.hour_labels = []

        # --- row header cells ---
        sale_hdr_rect = self.make_cell(self.label_w).move_to([label_x, sales_y, 0])
        sale_hdr = fit(Text("Hourly Sales", color=HEADER_C, weight=BOLD, font_size=30),
                       self.label_w * 0.85).move_to(sale_hdr_rect)
        hour_hdr_rect = self.make_cell(self.label_w).move_to([label_x, hour_y, 0])
        hour_hdr = fit(Text("Hour", color=HEADER_C, weight=BOLD, font_size=30),
                       self.label_w * 0.85).move_to(hour_hdr_rect)
        self.table.add(sale_hdr_rect, sale_hdr, hour_hdr_rect, hour_hdr)

        # --- data cells ---
        for j in range(n):
            # sales
            s_rect = self.make_cell(self.cell_w).move_to([data_x(j), sales_y, 0])
            is_oos = j in OOS_IDX
            s_col = OOS_C if is_oos else KNOWN_C
            s_lbl = fit(Text(SALES[j], color=s_col, weight=BOLD, font_size=32),
                        self.cell_w * 0.78).move_to(s_rect)
            self.sales_rects.append(s_rect)
            self.sales_labels.append(s_lbl)

            # hour
            h_rect = self.make_cell(self.cell_w).move_to([data_x(j), hour_y, 0])
            h_lbl = fit(Text(str(HOURS[j]), color=GREY_A, weight=BOLD, font_size=30),
                        self.cell_w * 0.78).move_to(h_rect)
            self.hour_rects.append(h_rect)
            self.hour_labels.append(h_lbl)

            self.table.add(s_rect, s_lbl, h_rect, h_lbl)

        self.table.shift(table_shift)
        return self.table

    def set_sales_value(self, j, text, color):
        """Return a new value mobject centred on sales cell j."""
        new = fit(Text(text, color=color, weight=BOLD, font_size=32),
                  self.cell_w * 0.78).move_to(self.sales_rects[j])
        return new


# ----------------------------------------------------------------------------
# Scene 1 — the data and the problem
# ----------------------------------------------------------------------------
class TableIntro(ImputationBase):
    def construct(self):
        title = Text("Demand Unconstraining", color=HEADER_C, weight=BOLD, font_size=44)
        title.to_edge(UP, buff=0.6)

        self.build_table(table_shift=DOWN * 0.3)

        self.play(Write(title), run_time=1.2)
        self.play(Create(self.table), run_time=2.2)
        self.wait(0.3)

        # highlight the OOS cells
        oos_boxes = VGroup(*[
            SurroundingRectangle(self.sales_rects[j], color=OOS_C, buff=0.0, stroke_width=5)
            for j in OOS_IDX
        ])
        self.play(LaggedStart(*[Create(b) for b in oos_boxes], lag_ratio=0.3), run_time=1.2)
        self.play(*[Indicate(self.sales_labels[j], color=OOS_C, scale_factor=1.3) for j in OOS_IDX],
                  run_time=1.0)

        caption = Text("Out of stock  \u2192  true demand is hidden",
                       color=OOS_C, weight=BOLD, font_size=30)
        caption.next_to(self.table, DOWN, buff=0.6)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=1.0)
        self.wait(1.5)


# ----------------------------------------------------------------------------
# Scene 2 — initialise with the 7-day mean
# ----------------------------------------------------------------------------
class InitialImpute(ImputationBase):
    def construct(self):
        step = Text("Step 1  \u2014  initialise with the 7-day mean",
                    color=MEAN_C, weight=BOLD, font_size=34)
        step.to_edge(UP, buff=0.6)

        self.build_table(table_shift=DOWN * 0.2)
        self.add(self.table)
        # keep OOS boxes visible
        oos_boxes = VGroup(*[
            SurroundingRectangle(self.sales_rects[j], color=OOS_C, buff=0.0, stroke_width=5)
            for j in OOS_IDX
        ])
        self.add(oos_boxes)
        self.play(Write(step), run_time=1.1)

        # small "last 7 days" strips feeding each OOS cell
        means = {X_COL: X_MEAN, Y_COL: Y_MEAN}
        samples = {X_COL: [9, 7, 8, 8, 9, 7, 8], Y_COL: [7, 6, 8, 7, 7, 6, 8]}

        for col in OOS_IDX:
            vals = samples[col]
            strip = VGroup(*[
                Text(str(v), color=GREY_B, weight=BOLD, font_size=24) for v in vals
            ]).arrange(RIGHT, buff=0.18)
            strip.next_to(self.sales_rects[col], UP, buff=1.4)
            brace = Brace(strip, DOWN, color=GREY_B)
            btxt = Text("last 7 days", color=GREY_B, font_size=22).next_to(brace, DOWN, buff=0.1)
            arrow = Arrow(btxt.get_bottom() + DOWN * 0.05,
                          self.sales_rects[col].get_top(),
                          color=MEAN_C, buff=0.1, stroke_width=5, max_tip_length_to_length_ratio=0.25)

            self.play(FadeIn(strip, shift=DOWN * 0.2), GrowFromCenter(brace),
                      FadeIn(btxt), run_time=0.9)
            self.play(GrowArrow(arrow), run_time=0.6)

            new_val = self.set_sales_value(col, fmt(means[col]), MEAN_C)
            self.play(
                ReplacementTransform(self.sales_labels[col], new_val),
                self.sales_rects[col].animate.set_stroke(MEAN_C),
                run_time=0.8,
            )
            self.sales_labels[col] = new_val
            self.play(FadeOut(strip), FadeOut(brace), FadeOut(btxt), FadeOut(arrow),
                      run_time=0.6)

        self.play(oos_boxes.animate.set_stroke(MEAN_C), run_time=0.5)
        note = Text("a rough starting guess", color=MEAN_C, font_size=28)
        note.next_to(self.table, DOWN, buff=0.6)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)


# ----------------------------------------------------------------------------
# Scene 3 — iterative imputation (3 rounds, Gauss-Seidel style)
# ----------------------------------------------------------------------------
class IterativeImpute(ImputationBase):
    def construct(self):
        title = Text("Step 2  \u2014  iterative imputation", color=PRED_C,
                     weight=BOLD, font_size=34).to_edge(UP, buff=0.45)

        self.build_table(table_shift=DOWN * 0.4)
        # start from the mean-initialised state (yellow)
        for col in OOS_IDX:
            init = X_MEAN if col == X_COL else Y_MEAN
            nv = self.set_sales_value(col, fmt(init), MEAN_C)
            self.sales_labels[col].become(nv)
            self.sales_rects[col].set_stroke(MEAN_C)

        self.add(self.table, title)

        round_lbl = Text("Round 0", color=GREY_A, weight=BOLD, font_size=30)
        round_lbl.to_corner(UL, buff=0.5)
        self.add(round_lbl)

        # a small legend for x / y
        xy_tag = VGroup(
            Text("x = hour 13", color=PRED_C, weight=BOLD, font_size=24),
            Text("y = hour 14", color=PRED_C, weight=BOLD, font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UR, buff=0.5)
        self.play(FadeIn(xy_tag), run_time=0.6)

        def predict_step(target_col, new_value, name):
            """Highlight the other sales cells, then update target_col."""
            feature_cols = [j for j in range(len(HOURS)) if j != target_col]
            feat_boxes = VGroup(*[
                SurroundingRectangle(self.sales_rects[j], color=FEATURE_C,
                                     buff=0.0, stroke_width=4)
                for j in feature_cols
            ])
            target_box = SurroundingRectangle(self.sales_rects[target_col],
                                              color=PRED_C, buff=0.0, stroke_width=5)

            # arrow + model tag coming from ABOVE the target cell (free space)
            model = VGroup(
                RoundedRectangle(width=1.7, height=0.7, corner_radius=0.15,
                                 stroke_color=MODEL_C, fill_color=BLACK, fill_opacity=1),
                Text("f( · )", color=MODEL_C, weight=BOLD, font_size=26),
            )
            model.move_to(self.sales_rects[target_col].get_top() + UP * 1.35)
            arrow = Arrow(model.get_bottom(), self.sales_rects[target_col].get_top(),
                          color=PRED_C, buff=0.1, stroke_width=5,
                          max_tip_length_to_length_ratio=0.35)
            sub = Text(f"predict {name}", color=PRED_C, font_size=24)
            sub.next_to(model, UP, buff=0.15)

            self.play(LaggedStart(*[Create(b) for b in feat_boxes], lag_ratio=0.08),
                      run_time=0.7)
            self.play(FadeIn(model, shift=DOWN * 0.15), FadeIn(sub), run_time=0.5)
            self.play(GrowArrow(arrow), Create(target_box), run_time=0.5)

            new_lbl = self.set_sales_value(target_col, fmt(new_value), PRED_C)
            self.play(
                ReplacementTransform(self.sales_labels[target_col], new_lbl),
                self.sales_rects[target_col].animate.set_stroke(PRED_C),
                Flash(self.sales_rects[target_col].get_center(), color=PRED_C,
                      line_length=0.25, num_lines=12, flash_radius=0.7),
                run_time=0.8,
            )
            self.sales_labels[target_col] = new_lbl
            self.play(FadeOut(feat_boxes), FadeOut(target_box),
                      FadeOut(model), FadeOut(arrow), FadeOut(sub), run_time=0.5)

        for r in range(1, 4):
            new_round = Text(f"Round {r}", color=YELLOW, weight=BOLD, font_size=30)
            new_round.to_corner(UL, buff=0.5)
            self.play(ReplacementTransform(round_lbl, new_round), run_time=0.4)
            round_lbl = new_round

            # predict y (hour 14) using x + others, then x (hour 13) using y + others
            predict_step(Y_COL, Y_SERIES[r], "y")
            predict_step(X_COL, X_SERIES[r], "x")
            self.wait(0.2)

        done = Text("estimates have stabilised", color=PRED_C, weight=BOLD, font_size=28)
        done.next_to(self.table, DOWN, buff=0.55)
        self.play(FadeIn(done, shift=UP * 0.2), run_time=0.8)
        self.wait(1.5)


# ----------------------------------------------------------------------------
# Scene 4 — convergence plot
# ----------------------------------------------------------------------------
class Convergence(ImputationBase):
    def construct(self):
        title = Text("Convergence over rounds", color=HEADER_C, weight=BOLD,
                     font_size=36).to_edge(UP, buff=0.6)

        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[6, 9, 1],
            x_length=8.5,
            y_length=4.5,
            axis_config={"color": GREY_B},
            tips=False,
        ).shift(DOWN * 0.4)

        # manual tick labels (Text/Pango -> no LaTeX dependency)
        x_ticks = VGroup(*[
            Text(str(r), color=GREY_A, font_size=24).next_to(axes.c2p(r, 6), DOWN, buff=0.2)
            for r in range(4)
        ])
        y_ticks = VGroup(*[
            Text(str(v), color=GREY_A, font_size=24).next_to(axes.c2p(0, v), LEFT, buff=0.2)
            for v in range(6, 10)
        ])
        x_axis_lbl = Text("round", color=GREY_A, font_size=26).next_to(x_ticks, DOWN, buff=0.25)
        y_axis_lbl = Text("value", color=GREY_A, font_size=26).next_to(y_ticks, LEFT, buff=0.3).rotate(PI / 2)

        self.play(Write(title), run_time=1.0)
        self.play(Create(axes), FadeIn(x_ticks), FadeIn(y_ticks),
                  FadeIn(x_axis_lbl), FadeIn(y_axis_lbl), run_time=1.5)

        rounds = [0, 1, 2, 3]
        x_pts = [axes.c2p(r, v) for r, v in zip(rounds, X_SERIES)]
        y_pts = [axes.c2p(r, v) for r, v in zip(rounds, Y_SERIES)]

        x_line = VMobject(color=PRED_C, stroke_width=4).set_points_as_corners(x_pts)
        y_line = VMobject(color="#4AA3FF", stroke_width=4).set_points_as_corners(y_pts)
        x_dots = VGroup(*[Dot(p, color=PRED_C, radius=0.07) for p in x_pts])
        y_dots = VGroup(*[Dot(p, color="#4AA3FF", radius=0.07) for p in y_pts])

        legend = VGroup(
            VGroup(Dot(color=PRED_C, radius=0.09),
                   Text("x = hour 13", color=PRED_C, font_size=24)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(color="#4AA3FF", radius=0.09),
                   Text("y = hour 14", color="#4AA3FF", font_size=24)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UR, buff=0.8).shift(DOWN * 0.5)

        self.play(FadeIn(legend), run_time=0.6)
        self.play(Create(x_line), Create(y_line), run_time=2.0)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in [*x_dots, *y_dots]],
                              lag_ratio=0.1), run_time=1.2)

        # dashed lines to the converged values
        xf = DashedLine(axes.c2p(0, X_SERIES[-1]), axes.c2p(3, X_SERIES[-1]),
                        color=PRED_C, stroke_width=2, dash_length=0.1).set_opacity(0.5)
        yf = DashedLine(axes.c2p(0, Y_SERIES[-1]), axes.c2p(3, Y_SERIES[-1]),
                        color="#4AA3FF", stroke_width=2, dash_length=0.1).set_opacity(0.5)
        self.play(Create(xf), Create(yf), run_time=0.8)
        self.wait(1.5)


# ----------------------------------------------------------------------------
# Full continuous version (all scenes chained)
# ----------------------------------------------------------------------------
class FullStory(ImputationBase):
    def construct(self):
        TableIntro.construct(self)
        self.clear()
        InitialImpute.construct(self)
        self.clear()
        IterativeImpute.construct(self)
        self.clear()
        Convergence.construct(self)
