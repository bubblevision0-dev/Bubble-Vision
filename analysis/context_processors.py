from .models import Teacher

def teacher_assignment(request):
    """
    Adds assigned_grade/assigned_section/assigned_subject
    for the logged-in user (Teacher) on ALL pages.
    """
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        return {}

    t = (
        Teacher.objects
        .select_related("grade", "section", "subject")
        .filter(user=request.user)
        .first()
    )

    return {
        "assigned_grade": getattr(t, "grade", None),
        "assigned_section": getattr(t, "section", None),
        "assigned_subject": getattr(t, "subject", None),
    }
