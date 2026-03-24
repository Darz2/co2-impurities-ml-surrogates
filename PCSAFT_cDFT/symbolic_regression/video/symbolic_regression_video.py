from manim import *
import json
import numpy as np

class SymbolicRegressionVideo(Scene):
    def construct(self):
        # --------------------------------
        # 1. Load exported Julia results
        # --------------------------------
        with open("sr_video_data.json", "r") as f:
            data = json.load(f)

        x_data = np.array(data["x_data"], dtype=float)
        y_data = np.array(data["y_data"], dtype=float)
        equations = data["equations"]
        best_equation = data["best_equation"]

        # --------------------------------
        # 2. Scene 1: Title
        # --------------------------------
        title = Text("Discovering an Equation from Data", font_size=40)
        subtitle = Text("Symbolic Regression", font_size=28).next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # --------------------------------
        # 3. Scene 2: Axes and data points
        # --------------------------------
        x_min, x_max = float(x_data.min()), float(x_data.max())
        y_min, y_max = float(y_data.min()), float(y_data.max())

        ax = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[min(0, y_min - 1), y_max + 1, 2],
            x_length=8,
            y_length=4.5,
            axis_config={"include_numbers": True},
        ).to_edge(DOWN)

        labels = ax.get_axis_labels(x_label="x", y_label="y")

        dots = VGroup(*[
            Dot(ax.c2p(x, y), radius=0.045)
            for x, y in zip(x_data, y_data)
        ])

        self.play(Create(ax), Write(labels))
        self.play(FadeIn(dots))
        self.wait(1)

        # --------------------------------
        # 4. Equation display area
        # --------------------------------
        eq_header = Text("Candidate equation:", font_size=28).to_edge(UP)
        eq_text = MathTex("f(x)=\\, ?", font_size=40).next_to(eq_header, DOWN)

        self.play(Write(eq_header), Write(eq_text))
        self.wait(0.5)

        # --------------------------------
        # 5. Animate discovered equations
        # --------------------------------
        current_curve = None

        for item in equations:
            eq_string = item["equation"]

            # very light cleanup for display
            eq_latex = eq_string.replace("*", r"\cdot ")

            new_eq = MathTex("f(x)=" + eq_latex, font_size=40).next_to(eq_header, DOWN)

            x_plot = np.array(item["x_plot"], dtype=float)
            y_plot = np.array(item["y_plot"], dtype=float)

            points = [ax.c2p(x, y) for x, y in zip(x_plot, y_plot)]
            new_curve = VMobject()
            new_curve.set_points_as_corners(points)

            if current_curve is None:
                self.play(
                    Transform(eq_text, new_eq),
                    Create(new_curve),
                    run_time=1.2
                )
            else:
                self.play(
                    Transform(eq_text, new_eq),
                    Transform(current_curve, new_curve),
                    run_time=1.2
                )

            current_curve = new_curve
            self.wait(0.7)

        # --------------------------------
        # 6. Final highlight
        # --------------------------------
        final_label = Text("Selected model", font_size=28, color=YELLOW).next_to(eq_text, DOWN)
        box = SurroundingRectangle(eq_text, buff=0.15, color=YELLOW)

        self.play(FadeIn(final_label), Create(box))
        self.wait(1.5)

        # --------------------------------
        # 7. Ending scene
        # --------------------------------
        final_eq = Text(f"Best equation: {best_equation}", font_size=26).to_edge(DOWN)
        self.play(FadeIn(final_eq))
        self.wait(2)