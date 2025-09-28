# report_generator.py (UPDATED to show metric_label & metric_value and improved chart)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
import os
import datetime
import tempfile
import traceback

def plot_summary(results, out_png):
    """
    Create a summary chart for the given results and save to out_png (full path).
    results: dict exercise -> {"metric_value": int, "eval": {"average_score": float}}
    """
    exercises = list(results.keys())
    # metric values (for chart): prefer metric_value; if missing, use reps
    metric_vals = [results[e].get("metric_value", results[e].get("reps", 0)) for e in exercises]
    scores = [results[e].get("eval", {}).get("average_score", 0) for e in exercises]

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    try:
        ax = axes[0]
        ax.bar(exercises, metric_vals, color="tab:blue")
        ax.set_title("Metric (reps / jumps / duration)")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel("Value")

        ax2 = axes[1]
        ax2.bar(exercises, scores, color="tab:green")
        ax2.set_ylim(0, 10)
        ax2.set_title("Avg Score (0-10)")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_png

def generate_pdf_report(user_info, results, filename=None):
    """
    user_info: dict with keys "name","age","gender","body_type","height","weight"
    results: dict: exercise -> {"metric_label": str, "metric_value": int, "reps": int, "eval": {...}}
    """
    safe_name = (user_info.get("name") or "user").strip().replace(" ", "_")
    if not safe_name:
        safe_name = "user"
    if filename is None:
        filename = f"{safe_name}_athletiq_report.pdf"
    filename = os.path.abspath(filename)

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title & user info
    story.append(Paragraph("Athletiq — Fitness Assessment Report", styles["Title"]))
    story.append(Spacer(1,12))

    ui_text = (
        f"Name: {user_info.get('name','-')}<br/>"
        f"Age: {user_info.get('age','-')}<br/>"
        f"Gender: {user_info.get('gender','-')}<br/>"
        f"Body Type: {user_info.get('body_type','-')}<br/>"
        f"Height (cm): {user_info.get('height','-')}    Weight (kg): {user_info.get('weight','-')}<br/>"
        f"Date: {datetime.date.today().isoformat()}"
    )
    story.append(Paragraph(ui_text, styles["Normal"]))
    story.append(Spacer(1,12))

    total_score = 0.0
    score_count = 0

    for ex, data in results.items():
        story.append(Paragraph(f"<b>{ex.capitalize()}</b>", styles["Heading2"]))
        metric_label = data.get("metric_label", "Reps")
        metric_value = data.get("metric_value", data.get("reps", 0))
        story.append(Paragraph(f"{metric_label}: {metric_value}", styles["Normal"]))
        story.append(Spacer(1,6))

        if "eval" in data and data["eval"].get("scores"):
            eval_data = data["eval"]
            table_data = [["Joint", "Angle", "Expected", "Score (0-10)"]]
            for joint, info in eval_data["scores"].items():
                angle_txt = str(info.get("angle", "-"))
                expected_txt = str(info.get("expected", "-"))
                score_txt = str(info.get("score", "-"))
                table_data.append([joint, angle_txt, expected_txt, score_txt])
            table = Table(table_data, hAlign="LEFT", colWidths=[120, 80, 140, 80])
            table.setStyle(TableStyle([
                ("GRID",(0,0),(-1,-1),0.5,colors.black),
                ("BACKGROUND",(0,0),(-1,0),colors.lightblue),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            story.append(table)
            story.append(Spacer(1,6))
            avg = eval_data.get("average_score", 0)
            story.append(Paragraph(f"Average Score (this exercise): {avg} / 10", styles["Normal"]))
            total_score += float(avg or 0)
            score_count += 1
        else:
            story.append(Paragraph("No detailed evaluation data available for this exercise.", styles["Normal"]))
        story.append(Spacer(1,12))

    # Chart: write to temp PNG and insert
    temp_chart = None
    if results:
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_chart = tf.name
            tf.close()
            plot_summary(results, temp_chart)
            story.append(Paragraph("<b>Performance Summary</b>", styles["Heading2"]))
            story.append(Spacer(1,6))
            story.append(Image(os.path.abspath(temp_chart), width=450, height=225))
            story.append(Spacer(1,12))
        except Exception:
            story.append(Paragraph("Warning: Could not generate performance chart.", styles["Normal"]))
            story.append(Paragraph("<pre>" + traceback.format_exc() + "</pre>", styles["Code"]))
            story.append(Spacer(1,12))

    overall = round((total_score / score_count), 2) if score_count > 0 else 0.0
    remark = "Excellent" if overall >= 8 else ("Good" if overall >= 5 else "Needs improvement")
    story.append(Paragraph("<b>Overall Score</b>", styles["Heading2"]))
    story.append(Paragraph(f"{overall} / 10 — {remark}", styles["Normal"]))
    story.append(Spacer(1,12))

    try:
        doc.build(story)
    finally:
        try:
            if temp_chart and os.path.exists(temp_chart):
                os.remove(temp_chart)
        except Exception:
            pass

    return filename
