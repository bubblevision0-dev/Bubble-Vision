from django import template

register = template.Library()

@register.filter
def is_scanned(matrix, args):
    """
    Usage: {% if scanned_matrix|is_scanned:lookup_string %}
    args should be "student_name,quiz_id"
    """
    if not matrix or not args:
        return False
    
    try:
        name, q_id = args.split(',')
        # Convert q_id to int because the matrix set has (str, int)
        return (name, int(q_id)) in matrix
    except (ValueError, TypeError):
        return False