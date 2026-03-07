# views.py (same file, add this below upload_bubble_image or near your other scan views)

import base64
import re
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
import os
import uuid
import cv2

from analysis.models import Grade, Section, Subject, AnswerKey
from .omr_sheet_detection import detect_and_warp_sheet_from_file


def capture_bubble_sheet(request, grade_id, section_id, subject_id):
    """
    Full-screen camera page with a green frame overlay.

    URL example:
        /capture/8/9/5/  -> grade_id=8, section_id=9, subject_id=5

    GET:
        - Show camera preview + "Capture" button.
    POST:
        - Receive base64 image_data from the browser.
        - Decode and save to a temp file.
        - Run detect_and_warp_sheet_from_file (4-corner boxes + zoom).
        - Save final cropped image under:
              MEDIA_ROOT/bubble_scans/<grade>/<section>/<subject>/sheet_<uuid>.png
        - Redirect back to answer_key_review for that grade/section/subject.
    """
    grade = get_object_or_404(Grade, pk=grade_id)
    section = get_object_or_404(Section, pk=section_id)
    subject = get_object_or_404(Subject, pk=subject_id)

    # Get the corresponding answer key (assumes one per grade+section+subject)
    answer_key = (
        AnswerKey.objects.filter(
            grade=grade,
            section=section,
            subject=subject,
        ).first()
    )

    if answer_key is None:
        messages.error(
            request,
            "No answer key found for this Grade / Section / Subject."
        )
        return redirect("home")  # adjust if your home url name is different

    if request.method == "POST":
        image_data = request.POST.get("image_data")
        if not image_data:
            messages.error(request, "No image data received from camera.")
            return redirect(
                "capture_bubble_sheet",
                grade_id=grade_id,
                section_id=section_id,
                subject_id=subject_id,
            )

        # image_data is "data:image/png;base64,AAAA..."
        match = re.match(
            r"^data:image/(?P<ext>png|jpeg|jpg);base64,(?P<data>.+)$",
            image_data,
        )
        if not match:
            messages.error(request, "Invalid image data format.")
            return redirect(
                "capture_bubble_sheet",
                grade_id=grade_id,
                section_id=section_id,
                subject_id=subject_id,
            )

        ext = match.group("ext")
        b64_data = match.group("data")

        try:
            img_bytes = base64.b64decode(b64_data)
        except Exception:
            messages.error(request, "Failed to decode image data.")
            return redirect(
                "capture_bubble_sheet",
                grade_id=grade_id,
                section_id=section_id,
                subject_id=subject_id,
            )

        # ----- 1) TEMP SAVE raw capture -----
        tmp_dir = os.path.join(settings.MEDIA_ROOT, "_tmp_scans")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_filename = f"{uuid.uuid4().hex}.{ext}"
        tmp_path = os.path.join(tmp_dir, tmp_filename)

        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        # ----- 2) Run detection + warp (4-corner boxes, or fallback center crop) -----
        try:
            final_img = detect_and_warp_sheet_from_file(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # ----- 3) SAVE under grade / section / subject -----
        def _safe(s):
            return str(s).strip().replace(" ", "_").replace("/", "_").lower()

        grade_folder = _safe(grade.name)
        section_folder = _safe(section.name)
        subject_folder = _safe(subject.name)

        scans_dir = os.path.join(
            settings.MEDIA_ROOT,
            "bubble_scans",
            grade_folder,
            section_folder,
            subject_folder,
        )
        os.makedirs(scans_dir, exist_ok=True)

        final_filename = f"sheet_{uuid.uuid4().hex}.png"
        final_path = os.path.join(scans_dir, final_filename)

        cv2.imwrite(final_path, final_img)

        messages.success(
            request,
            f"Bubble sheet captured and saved as {final_filename} for "
            f"{grade.name} - {section.name} - {subject.name}."
        )

        return redirect("answer_key_review", pk=answer_key.pk)

    # GET -> show camera page with overlay
    return render(
        request,
        "analysis/capture_bubble_sheet.html",
        {
            "grade": grade,
            "section": section,
            "subject": subject,
            "answer_key": answer_key,
        },
    )
