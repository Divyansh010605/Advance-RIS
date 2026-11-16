"""
ACR-Style PDF Report Generator for NIH ChestX-ray14 AI Service.
Generates structured, clinically-formatted radiology reports as PDFs.
Output format follows American College of Radiology (ACR) reporting standards.
"""
import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COLOR_HEADER_BG = (15, 23, 42)      # slate-900
COLOR_HEADER_TEXT = (248, 250, 252) # slate-50
COLOR_POSITIVE = (220, 38, 38)      # red-600
COLOR_NEGATIVE = (22, 163, 74)      # green-600
COLOR_UNCERTAIN = (234, 179, 8)     # yellow-500
COLOR_ACCENT = (59, 130, 246)       # blue-500
COLOR_LIGHT_BG = (241, 245, 249)    # slate-100
COLOR_BORDER = (203, 213, 225)      # slate-300


def generate_pdf_report(
    predictions: List[Dict[str, Any]],
    report_dict: Dict[str, Any],
    patient_id: str = "ANONYMOUS",
    study_date: Optional[str] = None,
    heatmap_b64: Optional[str] = None,
    ensemble_meta: Optional[Dict[str, Any]] = None,
    request_id: str = "UNKNOWN",
) -> bytes:
    """
    Generate an ACR-style PDF radiology report.

    Args:
        predictions: List of DiseasePrediction dicts (disease, probability, detected, uncertainty).
        report_dict: Report draft dict (findings, impression, detected_conditions).
        patient_id: DICOM Patient ID or 'ANONYMOUS' for de-identified cases.
        study_date: Study date string (YYYY-MM-DD). Defaults to today.
        heatmap_b64: Optional base64-encoded JPEG overlay image to embed.
        ensemble_meta: Optional ensemble metadata (models_evaluated, total_passes, etc.).
        request_id: API request ID for traceability.

    Returns:
        bytes: PDF file contents.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            HRFlowable,
            Image as RLImage,
            PageTemplate,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as e:
        raise RuntimeError(
            "reportlab is required for PDF generation. Install with: pip install reportlab"
        ) from e

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        fontName="Helvetica",
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica",
    )
    bold_style = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7,
        leading=10,
        textColor=colors.HexColor("#94a3b8"),
        fontName="Helvetica-Oblique",
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>ADVANCE-RIS CHEST RADIOLOGY REPORT</b>", ParagraphStyle(
                "HeaderTitle", parent=title_style, textColor=colors.white, fontSize=13, leading=16
            )),
            Paragraph(f"<b>REPORT ID:</b> {request_id[:16]}", ParagraphStyle(
                "HeaderMeta", parent=subtitle_style, textColor=colors.HexColor("#93c5fd"), alignment=2
            )),
        ],
        [
            Paragraph("NIH ChestX-ray14 AI-Assisted Screening System", ParagraphStyle(
                "HeaderSub", parent=subtitle_style, textColor=colors.HexColor("#cbd5e1"), fontSize=7.5
            )),
            Paragraph(f"Date: {study_date or datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", ParagraphStyle(
                "HeaderDate", parent=subtitle_style, textColor=colors.HexColor("#94a3b8"), alignment=2, fontSize=7.5
            )),
        ],
    ]
    header_table = Table(header_data, colWidths=["60%", "40%"])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4 * mm))

    # 2. Patient / Exam Information
    info_data = [
        [
            Paragraph("<b>Patient ID:</b>", body_style),
            Paragraph(patient_id, bold_style),
            Paragraph("<b>Modality:</b>", body_style),
            Paragraph("CR / DX (Chest Radiograph)", body_style),
        ],
        [
            Paragraph("<b>Study Date:</b>", body_style),
            Paragraph(study_date or datetime.utcnow().strftime("%Y-%m-%d"), body_style),
            Paragraph("<b>Projection:</b>", body_style),
            Paragraph("PA / AP Erect", body_style),
        ],
        [
            Paragraph("<b>AI Model:</b>", body_style),
            Paragraph(
                f"Tri-Model Ensemble ({', '.join(ensemble_meta.get('models_evaluated', ['densenet121'])) if ensemble_meta else 'DenseNet-121'})",
                body_style,
            ),
            Paragraph("<b>TTA Steps:</b>", body_style),
            Paragraph(str(ensemble_meta.get("tta_steps", 1) if ensemble_meta else 1), body_style),
        ],
    ]
    info_table = Table(info_data, colWidths=["20%", "30%", "20%", "30%"])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 4 * mm))

    # 3. Clinical Impression (Summary first, per ACR practice)
    story.append(Paragraph("IMPRESSION", h2_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3b82f6"), spaceAfter=4))

    impressions = report_dict.get("impression", [])
    detected = report_dict.get("detected_conditions", [])

    if not impressions or (len(detected) == 0):
        story.append(Paragraph(
            "<b>No acute cardiopulmonary abnormality detected.</b> Lungs are clear. Cardiomediastinal silhouette is within normal limits.",
            body_style,
        ))
    else:
        for i, imp in enumerate(impressions, 1):
            story.append(Paragraph(f"<b>{i}.</b> {imp}", body_style))

    story.append(Spacer(1, 4 * mm))

    # 4. Findings Table
    story.append(Paragraph("FINDINGS â€” 14-PATHOLOGY MULTI-LABEL SCREENING", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=4))

    table_data = [
        [
            Paragraph("<b>Pathology</b>", bold_style),
            Paragraph("<b>Probability</b>", bold_style),
            Paragraph("<b>Threshold</b>", bold_style),
            Paragraph("<b>Status</b>", bold_style),
            Paragraph("<b>Uncertainty</b>", bold_style),
        ]
    ]

    for p in predictions:
        disease = p.get("disease", "")
        if disease == "No_Finding":
            continue
        prob = p.get("probability", 0.0)
        thresh = p.get("threshold", 0.5)
        is_det = p.get("detected", False)
        unc = p.get("uncertainty")
        is_unc = p.get("is_uncertain", False)

        status_text = (
            "<font color='#dc2626'><b>DETECTED</b></font>"
            if is_det
            else "<font color='#16a34a'>Normal</font>"
        )
        if is_unc:
            status_text += " <font color='#ca8a04'>(Uncertain)</font>"

        unc_str = f"Â±{unc:.3f}" if unc is not None else "â€”"

        table_data.append([
            Paragraph(f"<b>{disease}</b>" if is_det else disease, body_style),
            Paragraph(f"{prob * 100:.1f}%", body_style),
            Paragraph(f"{thresh * 100:.0f}%", body_style),
            Paragraph(status_text, body_style),
            Paragraph(unc_str, body_style),
        ])

    path_table = Table(table_data, colWidths=["32%", "17%", "17%", "20%", "14%"])
    path_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for row_idx, p in enumerate(predictions, 1):
        if p.get("detected"):
            path_style_cmds.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fef2f2"))
            )
    path_table.setStyle(TableStyle(path_style_cmds))
    story.append(path_table)
    story.append(Spacer(1, 4 * mm))

    # 5. Narrative Findings (VLM text)
    story.append(Paragraph("NARRATIVE FINDINGS", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=4))
    findings = report_dict.get("findings", [])
    if findings:
        for f in findings:
            story.append(Paragraph(f"â€¢ {f}", body_style))
    else:
        story.append(Paragraph("Lungs are normally expanded and clear. No focal consolidation, pneumothorax, or large pleural effusion.", body_style))

    story.append(Spacer(1, 4 * mm))

    # 6. Heatmap Attachment
    if heatmap_b64:
        try:
            raw_img = base64.b64decode(heatmap_b64)
            img_buf = io.BytesIO(raw_img)
            story.append(Paragraph("LOCALIZATION â€” GRAD-CAM HEATMAP", h2_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=4))
            rl_img = RLImage(img_buf, width=65 * mm, height=65 * mm)
            story.append(rl_img)
            story.append(Spacer(1, 3 * mm))
        except Exception as e:
            logger.warning("Could not embed heatmap in PDF: %s", e)

    # 7. Disclaimer
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=3))
    story.append(Paragraph(
        "<b>RESEARCH &amp; SCREENING NOTICE:</b> This AI-generated report is produced by the Advance-RIS NIH ChestX-ray14 "
        "screening system (v2.2.0). It is intended as clinical decision support only and does NOT constitute a definitive "
        "medical diagnosis. All findings and impressions must be verified by a licensed radiologist or qualified physician.",
        disclaimer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


__all__ = ["generate_pdf_report"]
