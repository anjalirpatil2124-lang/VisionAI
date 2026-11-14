from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generate_detection_report(session, details):
    """
    Generate a PDF report for one VisionAI detection session.

    session format:
    (ID, Source, Total Objects, Average Confidence, Date)

    details format:
    [(Object Name, Confidence), ...]
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=colors.HexColor("#243B73"),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#243B73"),
        spaceBefore=14,
        spaceAfter=10
    )

    story = []

    # Report heading
    story.append(Paragraph("VisionAI", title_style))
    story.append(
        Paragraph("Object Detection Report", subtitle_style)
    )

    story.append(Spacer(1, 10))

    # Session information
    session_id = session[0]
    source = session[1]
    total_objects = session[2]
    average_confidence = session[3]
    session_date = session[4]

    story.append(
        Paragraph("Detection Session Information", heading_style)
    )

    session_data = [
        ["Session ID", str(session_id)],
        ["Detection Source", str(source)],
        ["Total Objects Detected", str(total_objects)],
        ["Average Confidence", f"{float(average_confidence):.2f}%"],
        ["Session Date", str(session_date)],
        ["Report Generated", datetime.now().strftime("%d-%m-%Y %H:%M")]
    ]

    session_table = Table(
        session_data,
        colWidths=[2.2 * inch, 3.8 * inch]
    )

    session_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1),
         colors.HexColor("#E8EEFC")),
        ("TEXTCOLOR", (0, 0), (-1, -1),
         colors.HexColor("#243B73")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    story.append(session_table)

    # Object details
    story.append(
        Paragraph("Detected Object Details", heading_style)
    )

    if details:
        table_data = [
            ["No.", "Object Name", "Confidence"]
        ]

        for index, item in enumerate(details, start=1):

            if isinstance(item, dict):
                object_name = item.get("name", "Unknown")
                confidence = item.get("confidence", 0)
            else:
                object_name = item[0]
                confidence = item[1]

            table_data.append([
                str(index),
                str(object_name).title(),
                f"{float(confidence):.2f}%"
            ])

        details_table = Table(
            table_data,
            colWidths=[0.7 * inch, 3.1 * inch, 2.2 * inch],
            repeatRows=1
        )

        details_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#4169E1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor("#F5F7FC")
            ]),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER")
        ]))

        story.append(details_table)

    else:
        story.append(
            Paragraph(
                "No individual object details are available.",
                styles["Normal"]
            )
        )

    # Footer
    story.append(Spacer(1, 30))

    story.append(
        Paragraph(
            "Generated by VisionAI - AI-Powered Object Detection System",
            subtitle_style
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()