"""
ACR-Style PDF Report Generator for NIH ChestX-ray14 AI Service.

Generates structured, clinically-formatted radiology reports as PDFs.
Output format is inspired by ACR (American College of Radiology) reporting standards.

Dependencies: reportlab (pure Python, no external binaries)
  pip install reportlab

Report sections:
  1. Header (Institution, Date/Time, Patient ID, Accession)
  2. Clinical Information
  3. Technique
  4. Findings (per-disease with probability and detection status)
  5. Impression (AI-generated summary)
  6. Disclaimer (mandatory AI-research disclaimer)
  7. Heatmap attachment (optional embedded Grad-CAM figure)
"""
import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Color palette (RGB tuples 0–255)
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
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as e:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        ) from e

    study_date_str = study_date or datetime.now().strftime("%Y-%m-%d")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    detected_diseases = [p for p in predictions if p.get("detected", False)]
    num_detected = len(detected_diseases)
    overall_impression = "ABNORMAL" if num_detected > 0 else "NORMAL"

    buf = io.BytesIO()
    W, H = A4

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.Color(*[c / 255 for c in COLOR_HEADER_TEXT]),
        spaceAfter=2,
    )
    style_subtitle = ParagraphStyle(
        "ReportSubtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.Color(*[c / 255 for c in COLOR_HEADER_TEXT]),
        spaceAfter=0,
    )
    style_section = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.Color(*[c / 255 for c in COLOR_ACCENT]),
        spaceBefore=8,
        spaceAfter=3,
    )
    style_body = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.black,
        leading=13,
        spaceAfter=2,
    )
    style_finding_pos = ParagraphStyle(
        "FindingPositive",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.Color(*[c / 255 for c in COLOR_POSITIVE]),
    )
    style_finding_neg = ParagraphStyle(
        "FindingNegative",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.Color(*[c / 255 for c in COLOR_NEGATIVE]),
    )
    style_disclaimer = ParagraphStyle(
        "Disclaimer",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.Color(0.5, 0.5, 0.5),
        leading=11,
        spaceBefore=6,
    )

    # ─── Build content flowables ──────────────────────────────────────────────
    story = []

    # ── Header block (dark band) ──────────────────────────────────────────────
    header_data = [
        [
            Paragraph("NIH ChestX-ray14 AI Screening Report", style_title),
            Paragraph(
                f"Patient ID: <b>{patient_id}</b><br/>"
                f"Study Date: {study_date_str}<br/>"
                f"Report Date: {generated_at}<br/>"
                f"Request ID: {request_id}",
                style_subtitle,
            ),
        ]
    ]
    header_table = Table(header_data, colWidths=[110 * mm, 70 * mm])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.Color(*[c / 255 for c in COLOR_HEADER_BG])),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 5 * mm))

    # ── Impression banner ─────────────────────────────────────────────────────
    imp_color = COLOR_POSITIVE if overall_impression == "ABNORMAL" else COLOR_NEGATIVE
    imp_data = [[
        Paragraph(
            f"OVERALL IMPRESSION: <b>{overall_impression}</b> "
            f"— {num_detected} finding(s) detected across 14 NIH disease categories",
            ParagraphStyle(
                "Impression",
                fontName="Helvetica-Bold",
                fontSize=10,
                textColor=colors.Color(*[c / 255 for c in imp_color]),
            ),
        )
    ]]
    imp_table = Table(imp_data, colWidths=[180 * mm])
    imp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.Color(*[c / 255 for c in COLOR_LIGHT_BG])),
                ("BOX", (0, 0), (-1, -1), 1.5, colors.Color(*[c / 255 for c in imp_color])),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(imp_table)
    story.append(Spacer(1, 4 * mm))

    # ── Technique ─────────────────────────────────────────────────────────────
    story.append(Paragraph("TECHNIQUE", style_section))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
    story.append(Spacer(1, 2 * mm))
    models_str = ", ".join(ensemble_meta.get("models_evaluated", ["DenseNet-121"])) if ensemble_meta else "DenseNet-121"
    tta_str = f"TTA ({ensemble_meta.get('total_passes', 1)} passes)" if ensemble_meta and ensemble_meta.get("tta_enabled") else "Single pass"
    story.append(Paragraph(
        f"Frontal chest radiograph processed at 224×224 px resolution. "
        f"AI models evaluated: {models_str}. Inference mode: {tta_str}. "
        f"Grad-CAM localization applied to detected conditions.",
        style_body,
    ))
    story.append(Spacer(1, 4 * mm))

    # ── Findings Table ────────────────────────────────────────────────────────
    story.append(Paragraph("FINDINGS", style_section))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
    story.append(Spacer(1, 2 * mm))

    table_data = [["Condition", "Probability", "Threshold", "Status", "Uncertainty"]]
    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(*[c / 255 for c in COLOR_HEADER_BG])),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(*[c / 255 for c in COLOR_LIGHT_BG])]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(*[c / 255 for c in COLOR_BORDER])),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    for i, pred in enumerate(predictions, start=1):
        detected = pred.get("detected", False)
        prob = pred.get("probability", 0.0)
        thresh = pred.get("threshold", 0.5)
        unc = pred.get("uncertainty")
        is_uncertain = pred.get("is_uncertain", False)

        status_str = "DETECTED" if detected else "NOT DETECTED"
        status_color = colors.Color(*[c / 255 for c in COLOR_POSITIVE]) if detected else colors.Color(*[c / 255 for c in COLOR_NEGATIVE])
        unc_str = f"{unc:.3f}" if unc is not None else "—"

        table_data.append([
            pred.get("disease", ""),
            f"{prob:.4f}",
            f"{thresh:.2f}",
            status_str,
            unc_str,
        ])

        if detected:
            table_styles.append(("TEXTCOLOR", (3, i), (3, i), status_color))
            table_styles.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        if is_uncertain:
            table_styles.append(("BACKGROUND", (4, i), (4, i), colors.Color(*[c / 255 for c in COLOR_UNCERTAIN])))

    findings_table = Table(
        table_data,
        colWidths=[55 * mm, 32 * mm, 28 * mm, 35 * mm, 30 * mm],
    )
    findings_table.setStyle(TableStyle(table_styles))
    story.append(findings_table)
    story.append(Spacer(1, 5 * mm))

    # ── AI Impression ─────────────────────────────────────────────────────────
    story.append(Paragraph("IMPRESSION", style_section))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
    story.append(Spacer(1, 2 * mm))
    for imp_line in report_dict.get("impression", ["No additional impression available."]):
        story.append(Paragraph(f"• {imp_line}", style_body))
    story.append(Spacer(1, 3 * mm))

    # ── Detailed Findings Narrative ───────────────────────────────────────────
    story.append(Paragraph("DETAILED NARRATIVE", style_section))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
    story.append(Spacer(1, 2 * mm))
    for finding_line in report_dict.get("findings", []):
        story.append(Paragraph(f"• {finding_line}", style_body))
    story.append(Spacer(1, 4 * mm))

    # ── Heatmap Figure ────────────────────────────────────────────────────────
    if heatmap_b64:
        try:
            story.append(Paragraph("GRAD-CAM LOCALIZATION", style_section))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
            story.append(Spacer(1, 2 * mm))
            hm_bytes = base64.b64decode(heatmap_b64)
            hm_buf = io.BytesIO(hm_bytes)
            img_flowable = RLImage(hm_buf, width=90 * mm, height=90 * mm)
            story.append(img_flowable)
            story.append(Paragraph(
                "Figure 1: Grad-CAM++ gradient-weighted class activation map overlaid on the original radiograph.",
                ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=8, textColor=colors.grey),
            ))
            story.append(Spacer(1, 4 * mm))
        except Exception as e:
            logger.warning("Could not embed heatmap in PDF: %s", e)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.Color(*[c / 255 for c in COLOR_BORDER])))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system trained on the NIH ChestX-ray14 research dataset "
        "and is intended exclusively for research, academic, and educational purposes. "
        "It has NOT been cleared or approved for clinical diagnostic use. "
        "All findings must be reviewed and validated by a qualified licensed radiologist before any clinical decision. "
        "The AI system may produce false positives and false negatives. "
        "Patient safety requires mandatory human clinical oversight.",
        style_disclaimer,
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="normal",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=frame)])
    doc.build(story)
    return buf.getvalue()
