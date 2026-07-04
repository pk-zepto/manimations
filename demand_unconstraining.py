"""
3Blue1Brown-style Manim animation explaining iterative imputation for
demand unconstraining of out-of-stock (OOS) hours.

Story:
  - Hourly sales with two OOS hours (13 = x, 14 = y): the true demand is hidden.
  - Step 1 (initialise): fill each OOS hour with the mean sales of the last 7 days.
  - Step 2 (iterate, 3 rounds): use x + ALL other hours to re-predict y; then use
    the fresh y + ALL other hours to re-predict x; repeat until stable.
  - Convergence: watch x and y settle over the rounds.

Render each scene separately (handy for individual PPT slides):
  manim -pqh demand_unconstraining.py TableIntro
  manim -pqh demand_unconstraining.py InitialImpute
  manim -pqh demand_unconstraining.py IterativeImpute
  manim -pqh demand_unconstraining.py Convergence

Or one continuous clip:
  manim -pqh demand_unconstraining.py FullStory

Tested with Manim Community v0.18+.  Everything uses the Courier New font
(no LaTeX required), so the whole thing stays in one clean monospace look.
"""

from manim import *

# ----------------------------------------------------------------------------
# Look & feel
# ----------------------------------------------------------------------------
MONO = "Courier New"

HEADER_C = "#8AB6E8"       # "Hourly Sales" / "Hour" headers  (light blue)
KNOWN_C = WHITE            # observed sales
OOS_C = "#E5484D"          # out-of-stock (censored) cells    (red)
MEAN_C = "#F1C40F"         # the 7-day mean initialisation     (yellow)
PRED_C = "#39D98A"         # freshly predicted / updated value (green)
FEATURE_C = "#3AC6D6"      # cells used as inputs to a prediction (cyan)
MODEL_C = "#BB86FC"        # the regression model f(.)         (purple)
VAR_C = "#FF79C6"          # the x / y variable tags           (pink)

# ----------------------------------------------------------------------------
# Problem data  (edit these to match your real numbers)
# ----------------------------------------------------------------------------
HOURS = [12, 13, 14, 15, 16, 17]
SALES = ["5", "OOS", "OOS", "4", "3", "1"]
OOS_IDX = [1, 2]
X_COL, Y_COL = 1, 2
X_MEAN, Y_MEAN = 8.0, 7.0

# Iterative-imputation trajectory: round 0 = init, then 3 refinement rounds
X_SERIES = [8.0, 7.4, 7.1, 7.0]
Y_SERIES = [7.0, 6.4, 6.7, 6.8]


def fit(mobj, max_w):
    if mobj.width > max_w:
        mobj.scale(max_w / mobj.width)
    return mobj


def fmt(v):
    return f"{v:.1f}"


class ImputationBase(Scene):
    label_w = 3.0
    cell_w = 1.15
    cell_h = 1.05
    table_shift = DOWN * 0.55

    # ---- table construction -------------------------------------------------
    def make_cell(self, w):
        return Rectangle(
            width=w, height=self.cell_h,
            stroke_color=WHITE, stroke_width=3,
            fill_color=BLACK, fill_opacity=1.0,
        )

    def build_table(self):
        n = len(HOURS)
        total_w = self.label_w + n * self.cell_w
        left = -total_w / 2.0
        sales_y = self.cell_h / 2.0
        hour_y = -self.cell_h / 2.0

        def data_x(j):
            return left + self.label_w + (j + 0.5) * self.cell_w

        label_x = left + self.label_w / 2.0

        self.table = VGroup()
        self.sales_rects, self.sales_labels = [], []
        self.hour_rects, self.hour_labels = [], []

        sale_hdr_rect = self.make_cell(self.label_w).move_to([label_x, sales_y, 0])
        sale_hdr = fit(Text("Hourly Sales", font=MONO, color=HEADER_C, weight=BOLD,
                            font_size=32), self.label_w * 0.86).move_to(sale_hdr_rect)
        hour_hdr_rect = self.make_cell(self.label_w).move_to([label_x, hour_y, 0])
        hour_hdr = fit(Text("Hour", font=MONO, color=HEADER_C, weight=BOLD,
                            font_size=32), self.label_w * 0.86).move_to(hour_hdr_rect)
        self.table.add(sale_hdr_rect, sale_hdr, hour_hdr_rect, hour_hdr)

        for j in range(n):
            s_rect = self.make_cell(self.cell_w).move_to([data_x(j), sales_y, 0])
            is_oos = j in OOS_IDX
            s_lbl = fit(Text(SALES[j], font=MONO, color=(OOS_C if is_oos else KNOWN_C),
                             weight=BOLD, font_size=34), self.cell_w * 0.72).move_to(s_rect)
            self.sales_rects.append(s_rect)
            self.sales_labels.append(s_lbl)

            h_rect = self.make_cell(self.cell_w).move_to([data_x(j), hour_y, 0])
            h_lbl = fit(Text(str(HOURS[j]), font=MONO, color=WHITE, weight=BOLD,
                             font_size=32), self.cell_w * 0.72).move_to(h_rect)
            self.hour_rects.append(h_rect)
            self.hour_labels.append(h_lbl)
            self.table.add(s_rect, s_lbl, h_rect, h_lbl)

        # crisp rounded outer border (matches the reference table)
        border = RoundedRectangle(
            width=total_w + 0.16, height=2 * self.cell_h + 0.16,
            corner_radius=0.16, stroke_color=WHITE, stroke_width=6,
        ).move_to(self.table)
        self.table.add(border)

        self.table.shift(self.table_shift)
        return self.table

    # ---- helpers ------------------------------------------------------------
    def set_sales_value(self, j, text, color):
        return fit(Text(text, font=MONO, color=color, weight=BOLD, font_size=34),
                   self.cell_w * 0.72).move_to(self.sales_rects[j])

    def preset_means(self):
        """Put the yellow mean values into the OOS cells (no animation)."""
        for col in OOS_IDX:
            val = X_MEAN if col == X_COL else Y_MEAN
            nv = self.set_sales_value(col, fmt(val), MEAN_C)
            self.sales_labels[col].become(nv)
            self.sales_rects[col].set_stroke(MEAN_C)

    def tag_pos(self, col):
        return self.sales_rects[col].get_top() + UP * 0.42

    def make_tag(self, col, letter):
        return Text(letter, font=MONO, color=VAR_C, weight=BOLD,
                    font_size=30).move_to(self.tag_pos(col))

    def add_tags_static(self):
        self.x_tag = self.make_tag(X_COL, "x")
        self.y_tag = self.make_tag(Y_COL, "y")
        self.add(self.x_tag, self.y_tag)

    # ---- prediction step (shared by IterativeImpute and FullStory) ----------
    def model_box(self):
        box = RoundedRectangle(width=2.1, height=0.95, corner_radius=0.16,
                               stroke_color=MODEL_C, fill_color=BLACK,
                               fill_opacity=1.0, stroke_width=4)
        lbl = Text("model", font=MONO, color=MODEL_C, weight=BOLD, font_size=30)
        grp = VGroup(box, lbl).move_to(UP * 2.45)
        return grp

    def predict_step(self, target_col, new_value):
        feature_cols = [j for j in range(len(HOURS)) if j != target_col]

        feat_boxes = VGroup(*[
            SurroundingRectangle(self.sales_rects[j], color=FEATURE_C, buff=0.0,
                                 stroke_width=4) for j in feature_cols
        ])
        target_box = SurroundingRectangle(self.sales_rects[target_col],
                                          color=PRED_C, buff=0.0, stroke_width=5)
        model = self.model_box()
        model_c = model[0].get_center()

        # 1) light up the input cells + the target cell
        self.play(LaggedStart(*[Create(b) for b in feat_boxes], lag_ratio=0.06),
                  run_time=0.6)
        self.play(FadeIn(model, shift=DOWN * 0.15), Create(target_box), run_time=0.5)

        # 2) copies of EVERY other hour's value stream up into the model -> the
        #    key intuition: the prediction uses ALL the other hours.
        copies = [self.sales_labels[j].copy() for j in feature_cols]
        self.play(
            LaggedStart(*[
                c.animate.move_to(model_c).scale(0.4).set_opacity(0.0)
                for c in copies
            ], lag_ratio=0.06),
            run_time=1.0,
        )
        for c in copies:
            self.remove(c)
        # pulse the model (scale + thicker border, no fill recolour)
        self.play(model.animate.scale(1.12).set_stroke(width=7),
                  rate_func=there_and_back, run_time=0.45)

        # 3) the model emits the new value, which flies down into the target cell
        new_lbl = self.set_sales_value(target_col, fmt(new_value), PRED_C)
        token = new_lbl.copy().move_to(model_c)
        self.play(FadeIn(token, scale=0.4), run_time=0.3)
        self.play(
            token.animate.move_to(self.sales_rects[target_col].get_center()),
            FadeOut(self.sales_labels[target_col]),
            self.sales_rects[target_col].animate.set_stroke(PRED_C),
            run_time=0.7,
        )
        self.remove(token)
        self.add(new_lbl)
        self.sales_labels[target_col] = new_lbl
        self.play(Flash(self.sales_rects[target_col].get_center(), color=PRED_C,
                        line_length=0.22, num_lines=12, flash_radius=0.6),
                  run_time=0.4)

        self.play(FadeOut(feat_boxes), FadeOut(target_box), FadeOut(model),
                  run_time=0.4)


# ----------------------------------------------------------------------------
# Scene 1 - the data, and a dynamic reveal of x and y
# ----------------------------------------------------------------------------
class TableIntro(ImputationBase):
    def construct(self):
        self.build_table()
        self.add(self.table)                       # first frame already shows the table

        # flag the two censored cells
        oos_boxes = VGroup(*[
            SurroundingRectangle(self.sales_rects[j], color=OOS_C, buff=0.0, stroke_width=6)
            for j in OOS_IDX
        ])
        self.play(LaggedStart(*[Create(b) for b in oos_boxes], lag_ratio=0.25), run_time=0.9)
        self.play(*[Indicate(self.sales_labels[j], color=OOS_C, scale_factor=1.35)
                    for j in OOS_IDX], run_time=0.9)
        self.wait(0.3)

        # dynamically bind a variable name to each censored cell:
        # a big letter grows out of the cell, then shrinks up to a small tag.
        for col, letter in [(X_COL, "x"), (Y_COL, "y")]:
            big = Text(letter, font=MONO, color=VAR_C, weight=BOLD,
                       font_size=90).move_to(self.sales_rects[col])
            tag = self.make_tag(col, letter)
            link = Line(self.sales_rects[col].get_top(), self.tag_pos(col),
                        color=VAR_C, stroke_width=3)
            self.play(FadeIn(big, scale=0.4), run_time=0.5)
            self.play(Transform(big, tag), GrowFromEdge(link, DOWN), run_time=0.7)
            self.play(Flash(self.tag_pos(col), color=VAR_C, line_length=0.15,
                            num_lines=10, flash_radius=0.35), run_time=0.4)
        self.wait(1.2)


# ----------------------------------------------------------------------------
# Scene 2 - initialise each OOS cell with the mean of the last 7 days
# ----------------------------------------------------------------------------
class InitialImpute(ImputationBase):
    def construct(self):
        self.build_table()
        self.add(self.table)
        self.add_tags_static()
        oos_boxes = VGroup(*[
            SurroundingRectangle(self.sales_rects[j], color=OOS_C, buff=0.0, stroke_width=6)
            for j in OOS_IDX
        ])
        self.add(oos_boxes)

        means = {X_COL: X_MEAN, Y_COL: Y_MEAN}
        samples = {X_COL: [9, 7, 8, 8, 9, 7, 8], Y_COL: [7, 6, 8, 7, 7, 6, 8]}

        for k, col in enumerate(OOS_IDX):
            strip = VGroup(*[
                Text(str(v), font=MONO, color=GREY_B, weight=BOLD, font_size=26)
                for v in samples[col]
            ]).arrange(RIGHT, buff=0.16)
            strip.next_to(self.sales_rects[col], UP, buff=1.55)
            brace = Brace(strip, DOWN, color=GREY_B)
            btxt = Text("last 7 days", font=MONO, color=GREY_B, font_size=24)
            btxt.next_to(brace, DOWN, buff=0.12)
            mean_tok = Text(fmt(means[col]), font=MONO, color=MEAN_C, weight=BOLD,
                            font_size=30).move_to(strip)

            self.play(FadeIn(strip, shift=DOWN * 0.2), GrowFromCenter(brace),
                      FadeIn(btxt), run_time=0.8)
            # collapse the 7 samples into their mean value...
            self.play(ReplacementTransform(strip, mean_tok),
                      FadeOut(brace), FadeOut(btxt), run_time=0.7)
            # ...then drop that mean into the cell
            new_val = self.set_sales_value(col, fmt(means[col]), MEAN_C)
            self.play(mean_tok.animate.move_to(self.sales_rects[col]),
                      FadeOut(self.sales_labels[col]),
                      self.sales_rects[col].animate.set_stroke(MEAN_C),
                      run_time=0.7)
            self.remove(mean_tok)
            self.add(new_val)
            self.sales_labels[col] = new_val
            self.play(Flash(self.sales_rects[col].get_center(), color=MEAN_C,
                            line_length=0.2, num_lines=12, flash_radius=0.55),
                      run_time=0.4)

        self.play(oos_boxes.animate.set_stroke(MEAN_C), run_time=0.4)
        self.wait(1.2)


# ----------------------------------------------------------------------------
# Scene 3 - iterative imputation (3 rounds); ALL other hours feed each prediction
# ----------------------------------------------------------------------------
class IterativeImpute(ImputationBase):
    def construct(self):
        self.build_table()
        self.preset_means()
        self.add(self.table)
        self.add_tags_static()

        # dynamic round counter (top-left, minimal)
        self.round_lbl = Text("round 1", font=MONO, color=GREY_A, weight=BOLD,
                              font_size=30).to_corner(UL, buff=0.5)
        self.add(self.round_lbl)

        for r in range(1, 4):
            if r > 1:
                new_lbl = Text(f"round {r}", font=MONO, color=GREY_A, weight=BOLD,
                               font_size=30).to_corner(UL, buff=0.5)
                self.play(ReplacementTransform(self.round_lbl, new_lbl), run_time=0.4)
                self.round_lbl = new_lbl
            # predict y from (x + other hours), then x from (fresh y + other hours)
            self.predict_step(Y_COL, Y_SERIES[r])
            self.predict_step(X_COL, X_SERIES[r])
            self.wait(0.15)
        self.wait(1.0)


# ----------------------------------------------------------------------------
# Scene 4 - convergence over the rounds (labels ride the line ends -> dynamic)
# ----------------------------------------------------------------------------
class Convergence(ImputationBase):
    def construct(self):
        axes = Axes(
            x_range=[0, 3, 1], y_range=[6, 9, 1],
            x_length=8.5, y_length=4.6,
            axis_config={"color": GREY_B, "stroke_width": 3},
            tips=False,
        ).shift(DOWN * 0.3 + RIGHT * 0.3)

        x_ticks = VGroup(*[
            Text(str(r), font=MONO, color=GREY_A, font_size=24)
            .next_to(axes.c2p(r, 6), DOWN, buff=0.2) for r in range(4)
        ])
        y_ticks = VGroup(*[
            Text(str(v), font=MONO, color=GREY_A, font_size=24)
            .next_to(axes.c2p(0, v), LEFT, buff=0.2) for v in range(6, 10)
        ])
        x_axis_lbl = Text("round", font=MONO, color=GREY_A, font_size=26)
        x_axis_lbl.next_to(x_ticks, DOWN, buff=0.3)

        self.play(Create(axes), FadeIn(x_ticks), FadeIn(y_ticks),
                  FadeIn(x_axis_lbl), run_time=1.3)

        rounds = [0, 1, 2, 3]
        x_pts = [axes.c2p(r, v) for r, v in zip(rounds, X_SERIES)]
        y_pts = [axes.c2p(r, v) for r, v in zip(rounds, Y_SERIES)]

        x_line = VMobject(color=PRED_C, stroke_width=5).set_points_as_corners(x_pts)
        y_line = VMobject(color="#4AA3FF", stroke_width=5).set_points_as_corners(y_pts)
        x_verts = VGroup(*[Dot(p, color=PRED_C, radius=0.055) for p in x_pts])
        y_verts = VGroup(*[Dot(p, color="#4AA3FF", radius=0.055) for p in y_pts])

        x_dot = Dot(x_pts[0], color=PRED_C, radius=0.1)
        y_dot = Dot(y_pts[0], color="#4AA3FF", radius=0.1)
        x_tag = Text("x", font=MONO, color=PRED_C, weight=BOLD, font_size=32)
        y_tag = Text("y", font=MONO, color="#4AA3FF", weight=BOLD, font_size=32)
        x_tag.add_updater(lambda m: m.next_to(x_dot, UR, buff=0.12))
        y_tag.add_updater(lambda m: m.next_to(y_dot, DR, buff=0.12))

        self.play(FadeIn(x_dot), FadeIn(y_dot), FadeIn(x_tag), FadeIn(y_tag),
                  run_time=0.5)

        # draw both series left->right, dots (and their tags) ride the drawn tip
        self.play(
            Create(x_line), Create(y_line),
            UpdateFromFunc(x_dot, lambda m: m.move_to(x_line.get_end())),
            UpdateFromFunc(y_dot, lambda m: m.move_to(y_line.get_end())),
            run_time=2.4, rate_func=linear,
        )
        self.play(FadeIn(x_verts), FadeIn(y_verts), run_time=0.4)
        x_tag.clear_updaters()
        y_tag.clear_updaters()

        xf = DashedLine(axes.c2p(0, X_SERIES[-1]), axes.c2p(3, X_SERIES[-1]),
                        color=PRED_C, stroke_width=2).set_opacity(0.45)
        yf = DashedLine(axes.c2p(0, Y_SERIES[-1]), axes.c2p(3, Y_SERIES[-1]),
                        color="#4AA3FF", stroke_width=2).set_opacity(0.45)
        self.play(Create(xf), Create(yf), run_time=0.7)
        self.play(Indicate(x_dot, color=PRED_C), Indicate(y_dot, color="#4AA3FF"),
                  run_time=0.6)
        self.wait(1.2)


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
