from django.urls import path
from . import views

urlpatterns = [
    
    path('super_admin_login/', views.super_admin_login, name='super_admin_login'),
    path('super_admin_logout/', views.super_admin_logout, name='super_admin_logout'),

    # Super Admin Dashboard
    path('institution-admin-dashboard/', views.institution_admin_dashboard, name='institution_admin_dashboard'),

    # Institution Management (Super Admin actions)
    path('add_institution/', views.add_institution, name='add_institution'),
    path('institution/<int:institution_id>/', views.institution_detail, name='institution_detail'),
    path('edit_institution/<int:institution_id>/', views.create_or_edit_institution, name='edit_institution'),
    path('delete_institution/<int:institution_id>/', views.delete_institution, name='delete_institution'),

    # User Management (Super Admin actions)
    path('create_user/', views.create_or_edit_user, name='create_user'),
    path('edit_user/<int:user_id>/', views.create_or_edit_user, name='edit_user'),
    path('delete_user/<int:user_id>/', views.delete_user, name='delete_user'),

    # Assign Institution Admin (Super Admin actions)
    path('assign_institution_admin/', views.assign_institution_admin, name='assign_institution_admin'),
    path('edit_institution_admin/<int:admin_id>/', views.assign_institution_admin, name='edit_institution_admin'),
    path('delete_institution_admin/<int:admin_id>/', views.delete_institution_admin, name='delete_institution_admin'),
    # ================== ADMIN LOGIN & DASHBOARD ==================
    path("login/", views.admin_login, name="admin_login"),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('update-sy/', views.update_school_year, name='update_school_year'),
    path('history/', views.sy_history_list, name='sy_history_list'),
    path('history/<str:year_slug>/', views.sy_history_detail, name='sy_history_detail'),
    path('toggle-user-status/<int:user_id>/', views.toggle_user_status, name='toggle_user_status'),
        
    path("institution/<int:institution_id>/dashboard/", views.institution_dashboard, name="institution_dashboard"),
    path("admin-logout/", views.admin_logout, name="admin_logout"),
    path("admin-profile/", views.admin_profile, name="admin_profile"),

    # ================== USER MANAGEMENT (ADMIN) ==================
    path("user-management/", views.user_management, name="user_management"),
    path("edit-user/<int:user_id>/", views.edit_user, name="edit_user"),
    path("delete-user/<int:user_id>/", views.delete_user, name="delete_user"),
    path("get_sections/<int:grade_id>/", views.get_sections, name="get_sections"),
    path("get_subjects/<int:grade_id>/", views.get_subjects, name="get_subjects"),
    
    path("get_available_subjects/<int:grade_id>/<int:section_id>/", views.get_available_subjects, name="get_available_subjects"),


    path("u/archives/years/", views.u_archives_years, name="u_archives_years"),
    path("u/archives/<str:sy>/grades/", views.u_archives_grades, name="u_archives_grades"),
    path("u/archives/<str:sy>/grades/<int:grade_id>/sections/", views.u_archives_sections, name="u_archives_sections"),
    path("u/archives/<str:sy>/grades/<int:grade_id>/sections/<int:section_id>/subjects/", views.u_archives_subjects, name="u_archives_subjects"),
    path("u/archives/<str:sy>/grades/<int:grade_id>/sections/<int:section_id>/subjects/<int:subject_id>/detail/",
         views.u_archives_subject_detail, name="u_archives_subject_detail"),
    
    path("school-year/<int:sy_id>/activate/", views.activate_school_year, name="activate_school_year"),
    path("school-year/<int:sy_id>/deactivate/", views.deactivate_school_year, name="deactivate_school_year"),

    # ================== GRADE CRUD ==================
    path("add-grade/", views.add_grade, name="add_grade"),
    path('add-grading-period/', views.add_grading_period, name='add_grading_period'),
    path('delete-grading-period/', views.delete_grading_period, name='delete_grading_period'), 
    path("grades/<int:grade_id>/edit/", views.edit_grade, name="edit_grade"),
    path("grades/<int:grade_id>/delete/", views.delete_grade, name="delete_grade"),
    path("grades/<int:grade_id>/", views.manage_grade, name="manage_grade"),

        # ================== STUDENT CRUD (ADMIN) ==================
    path("students/", views.student_list, name="student_list"),
    path(
        "students/import/",
        views.import_students,
        name="import_students",
    ),
    path("students/<int:pk>/delete/", views.delete_student, name="delete_student"),
    
    path("grades/<int:grade_id>/sections/add/", views.add_section, name="add_section"),
    path("grades/<int:grade_id>/sections/<int:section_id>/edit/", views.edit_section, name="edit_section"),
    path("grades/<int:grade_id>/sections/<int:section_id>/delete/", views.delete_section, name="delete_section"),

    path("grades/<int:grade_id>/subjects/add/", views.add_subject, name="add_subject"),
    path("grades/<int:grade_id>/subjects/<int:subject_id>/edit/", views.edit_subject, name="edit_subject"),
    path("grades/<int:grade_id>/subjects/<int:subject_id>/delete/", views.delete_subject, name="delete_subject"),
    
    path("user/tally-per-item/<int:answer_key_id>/", views.user_tally_per_item, name="user_tally_per_item"),

    # ================== TEACHER LOGIN & DASHBOARD ==================
    path("", views.user_login, name="user_login"),
    path("logout/", views.user_logout, name="user_logout"),
    path("dashboard/", views.user_dashboard, name="user_dashboard"),
    path("profile/", views.user_profile, name="user_profile"),

    # ================== TEACHER – SCAN HISTORY PAGE ==================
    path(
        "capture/<int:grade_id>/<int:section_id>/<int:subject_id>/",
        views.user_capture,
        name="user_capture",
    ),

    # ================== BUBBLE SHEET SCAN (UPLOAD/CAPTURE) ==========
    path('scan/<int:pk>/', views.scan_document, name='scan_document'),

    # ================== ANSWER KEY / REVIEW =========================
    path("users/uploads/", views.upload_answer_key, name="upload_answer_key"),
    path("users/answer-key/<int:pk>/review/", views.answer_key_review, name="answer_key_review"),
    path("get_sections/<int:grade_id>/", views.get_sections, name="get_sections"),
    path("get_subjects/<int:grade_id>/", views.get_subjects, name="get_subjects"),
    
    path("answer-key/<int:pk>/bubble-sheet/", views.download_bubble_sheets_for_answer_key, name="download_bubble_sheets_for_answer_key"),


    path(
        "convert-all-answer-keys/",
        views.convert_all_answer_keys,
        name="convert_all_answer_keys",
    ),
    path(
        "answer-key-review/<int:pk>/",
        views.answer_key_review,
        name="answer_key_review",
    ),
    path(
        "answer-key/<int:pk>/",
        views.view_answer_key,
        name="view_answer_key",
    ),
    path(
        "answer-key/<int:pk>/edit/",
        views.edit_answer_key,
        name="edit_answer_key",
    ),
    path(
        "answer-key/<int:pk>/delete/",
        views.delete_answer_key,
        name="delete_answer_key",
    ),

    # ================== AJAX – DROPDOWNS ============================
    path(
        "api/sections/<int:grade_id>/",
        views.get_sections,
        name="api_sections",
    ),
    path(
        "api/subjects/<int:grade_id>/",
        views.get_subjects,
        name="api_subjects",
    ),

    # ================== REPORTS & SAVED KEYS ========================
    path("users/report/", views.user_report, name="user_report"),
    path(
        "saved-answer-keys/",
        views.list_saved_answer_keys,
        name="saved_answer_keys",
    ),

    # ================== NEW: VIEW ONE SCAN RESULT ===================
    path(
        "scan-result/<int:pk>/",
        views.view_scan_result,
        name="view_scan_result",
    ),
    path("scan/modal/<int:scan_id>/", views.scan_result_modal, name="scan_result_modal"),


    path("delete-scan/<int:pk>/", views.delete_scan_result, name="delete_scan_result"),


    path("grades/<int:grade_id>/sections/<int:section_id>/edit/",  views.edit_section,  name="edit_section"),
    path("grades/<int:grade_id>/sections/<int:section_id>/delete/", views.delete_section, name="delete_section"),

    path("grades/<int:grade_id>/subjects/<int:subject_id>/edit/",  views.edit_subject,  name="edit_subject"),
    path("grades/<int:grade_id>/subjects/<int:subject_id>/delete/", views.delete_subject, name="delete_subject"),
    path("item-analysis/<int:answer_key_id>/", views.item_analysis, name="item_analysis"),
    path(
        "revise-difficult-items/<int:answer_key_id>/",
        views.revise_difficult_items,
        name="revise_difficult_items",
    ),
    path('get_sections/<int:grade_id>/', views.get_sections, name='get_sections'),
    path('get_subjects/<int:grade_id>/', views.get_subjects, name='get_subjects'),

    path("super-admin/delete-admin/<int:user_id>/", views.superadmin_delete_admin_user, name="superadmin_delete_admin_user"),
    path("add_institution/", views.add_institution, name="add_institution"),
    path("add_institution/<int:institution_id>/edit/", views.add_institution, name="superadmin_edit_institution"),
    path("add_institution/<int:institution_id>/delete/", views.superadmin_delete_institution, name="superadmin_delete_institution"),

    path(
        "my-class/<int:grade_id>/<int:section_id>/<int:subject_id>/students/",
        views.teacher_students,
        name="teacher_students",
    ),
    
    path("users/students_view/", views.students_view, name="students_view"),
    path("users/import/", views.import_students_view, name="import_students_view"),
    path("users/<int:student_id>/delete/", views.delete_student, name="user_delete_student"),
    path("sections/<int:section_id>/bubble-sheets/download/", views.download_bubble_sheets_selected,name="download_bubble_sheets_section"),

    path("add-grading-period/", views.add_grading_period, name="add_grading_period"),
    path("delete-grading-period/", views.delete_grading_period, name="delete_grading_period"),
]

