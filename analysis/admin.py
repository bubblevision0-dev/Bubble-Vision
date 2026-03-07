from django.contrib import admin
from django.contrib.auth.models import User
from django.db.models import Q

from .models import (
    Institution,
    InstitutionAdmin,
    InstitutionDetails,
    SchoolYear,
    CustomUser,
    GradingPeriod,
    Grade,
    Section,
    Subject,
    Student,
    Teacher,
    TeacherClassAssignment,
    AnswerKey,
    ScanResult,
    UserProfile,
    ClassAssignment,
    AssessmentRecord,
)

# ------------------------------------------------------------
# INLINES
# ------------------------------------------------------------

class InstitutionDetailsInline(admin.StackedInline):
    model = InstitutionDetails
    extra = 0


class SchoolYearInline(admin.TabularInline):
    model = SchoolYear
    extra = 0
    fields = ("year", "start_date", "end_date", "is_active")
    ordering = ("-year",)


class GradingPeriodInline(admin.TabularInline):
    model = GradingPeriod
    extra = 0
    fields = ("school_year", "period", "start_date", "end_date")
    ordering = ("school_year", "period")


class SectionInline(admin.TabularInline):
    model = Section
    extra = 0
    fields = ("name",)
    ordering = ("name",)


class SubjectInline(admin.TabularInline):
    model = Subject
    extra = 0
    fields = ("name", "description")
    ordering = ("name",)


class TeacherClassAssignmentInline(admin.TabularInline):
    model = TeacherClassAssignment
    extra = 0
    autocomplete_fields = ("grade", "section", "subject", "institution")
    fields = ("institution", "school_year", "grade", "section", "subject", "created_at")
    readonly_fields = ("created_at",)


# ------------------------------------------------------------
# MAIN ADMINS
# ------------------------------------------------------------

@admin.register(Institution)
class InstitutionAdminSite(admin.ModelAdmin):
    list_display = ("name", "address", "school_year")
    search_fields = ("name", "address", "school_year")
    ordering = ("name",)
    inlines = [InstitutionDetailsInline, SchoolYearInline, GradingPeriodInline]


@admin.register(InstitutionAdmin)
class InstitutionAdminAdmin(admin.ModelAdmin):
    list_display = ("user", "institution", "is_active")
    list_filter = ("is_active", "institution")
    search_fields = ("user__username", "user__first_name", "user__last_name", "institution__name")
    autocomplete_fields = ("user", "institution")


@admin.register(InstitutionDetails)
class InstitutionDetailsAdmin(admin.ModelAdmin):
    list_display = ("institution",)
    search_fields = ("institution__name",)
    autocomplete_fields = ("institution",)


@admin.register(SchoolYear)
class SchoolYearAdmin(admin.ModelAdmin):
    list_display = ("year", "institution", "start_date", "end_date", "is_active")
    list_filter = ("is_active", "institution")
    search_fields = ("year", "institution__name")
    autocomplete_fields = ("institution",)
    ordering = ("-year",)


@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ("user", "institution")
    search_fields = ("user__username", "user__first_name", "user__last_name", "institution__name")
    autocomplete_fields = ("user", "institution")


@admin.register(GradingPeriod)
class GradingPeriodAdmin(admin.ModelAdmin):
    list_display = ("period", "school_year", "institution", "start_date", "end_date")
    list_filter = ("institution", "school_year", "period")
    search_fields = ("institution__name", "school_year", "period")
    autocomplete_fields = ("institution",)
    ordering = ("institution__name", "school_year", "period")


@admin.register(Grade)
class GradeAdmin(admin.ModelAdmin):
    list_display = ("name", "number", "institution", "school_year", "grading_period")
    list_filter = ("institution", "school_year", "grading_period")
    search_fields = ("name", "institution__name")
    ordering = ("number", "name")
    autocomplete_fields = ("institution", "school_year", "grading_period")
    inlines = [SectionInline, SubjectInline]


@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ("name", "grade", "get_institution")
    list_filter = ("grade__institution", "grade")
    search_fields = ("name", "grade__name", "grade__institution__name")
    autocomplete_fields = ("grade",)

    def get_institution(self, obj):
        return obj.grade.institution if obj.grade else None
    get_institution.short_description = "Institution"


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("name", "grade", "get_institution")
    list_filter = ("grade__institution", "grade")
    search_fields = ("name", "grade__name", "grade__institution__name")
    autocomplete_fields = ("grade",)

    def get_institution(self, obj):
        return obj.grade.institution if obj.grade else None
    get_institution.short_description = "Institution"


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("formatted_name", "institution", "school_year", "grade", "section")
    list_filter = ("institution", "school_year", "grade", "section")
    search_fields = ("full_name", "last_name", "middle_initial", "section__name", "grade__name", "institution__name")
    autocomplete_fields = ("institution", "grade", "section")
    ordering = ("last_name", "full_name")

    # optional safety: don't let people manually edit these if you want them always auto-synced
    # readonly_fields = ("institution", "school_year", "grade")


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display = ("user", "institution", "school_year", "grade", "section", "subject")
    list_filter = ("institution", "school_year")
    search_fields = ("user__username", "user__first_name", "user__last_name", "institution__name")
    autocomplete_fields = ("user", "institution", "grade", "section", "subject")
    inlines = [TeacherClassAssignmentInline]

    def save_model(self, request, obj, form, change):
        # auto-fill school_year from institution if missing (matches your model save)
        if obj.institution and not obj.school_year:
            obj.school_year = obj.institution.school_year
        super().save_model(request, obj, form, change)


@admin.register(TeacherClassAssignment)
class TeacherClassAssignmentAdmin(admin.ModelAdmin):
    list_display = ("teacher", "institution", "school_year", "grade", "section", "subject", "created_at")
    list_filter = ("institution", "school_year", "grade", "section", "subject")
    search_fields = (
        "teacher__user__username",
        "teacher__user__first_name",
        "teacher__user__last_name",
        "institution__name",
        "grade__name",
        "section__name",
        "subject__name",
    )
    autocomplete_fields = ("teacher", "institution", "grade", "section", "subject")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)


@admin.register(AnswerKey)
class AnswerKeyAdmin(admin.ModelAdmin):
    list_display = ("quiz_name", "institution", "school_year", "grading_period", "grade", "section", "subject", "user", "uploaded_at")
    list_filter = ("institution", "school_year", "grading_period", "grade", "section", "subject")
    search_fields = (
        "quiz_name",
        "user__username",
        "user__first_name",
        "user__last_name",
        "institution__name",
        "grade__name",
        "section__name",
        "subject__name",
    )
    autocomplete_fields = ("user", "institution", "grading_period", "grade", "section", "subject")
    readonly_fields = ("uploaded_at", "school_year", "total_points", "total_items")

    def save_model(self, request, obj, form, change):
        # auto-copy school_year from institution (matches your model save)
        if obj.institution and not obj.school_year:
            obj.school_year = obj.institution.school_year
        super().save_model(request, obj, form, change)


@admin.register(ScanResult)
class ScanResultAdmin(admin.ModelAdmin):
    list_display = ("student_name", "institution", "school_year", "grading_period", "grade", "section", "subject", "score", "max_score", "created_at", "is_active")
    list_filter = ("institution", "school_year", "grading_period", "grade", "section", "subject", "is_active")
    search_fields = (
        "student_name",
        "institution__name",
        "grade__name",
        "section__name",
        "subject__name",
        "answer_key__quiz_name",
        "answer_key__user__username",
    )
    autocomplete_fields = ("answer_key", "institution", "grading_period", "grade", "section", "subject", "student")
    readonly_fields = ("created_at", "institution", "school_year", "grading_period", "grade", "section", "subject")

    def save_model(self, request, obj, form, change):
        # keep consistent with your model save
        if obj.answer_key:
            obj.institution = obj.answer_key.institution
            obj.school_year = obj.answer_key.school_year
            obj.grading_period = obj.answer_key.grading_period
            obj.grade = obj.answer_key.grade
            obj.section = obj.answer_key.section
            obj.subject = obj.answer_key.subject
        super().save_model(request, obj, form, change)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "created_by")
    search_fields = ("user__username", "created_by__username")
    autocomplete_fields = ("user", "created_by")

    def save_model(self, request, obj, form, change):
        # auto-set created_by when created from admin
        if not change and not obj.created_by:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(ClassAssignment)
class ClassAssignmentAdmin(admin.ModelAdmin):
    list_display = ("teacher", "institution", "grade", "section", "subject", "grading_period")
    list_filter = ("institution", "grade", "section", "subject", "grading_period")
    search_fields = ("teacher__username", "institution__name", "grade__name", "section__name", "subject__name")
    autocomplete_fields = ("teacher", "institution", "grade", "section", "subject", "grading_period")


@admin.register(AssessmentRecord)
class AssessmentRecordAdmin(admin.ModelAdmin):
    list_display = ("institution", "school_year", "grade", "section", "subject", "teacher", "student", "score", "max_score", "assessment_name", "created_at")
    list_filter = ("institution", "school_year", "grade", "section", "subject", "teacher")
    search_fields = (
        "institution__name",
        "school_year",
        "assessment_name",
        "student__full_name",
        "student__last_name",
        "teacher__user__username",
    )
    autocomplete_fields = ("institution", "grade", "section", "subject", "teacher", "student")
    readonly_fields = ("created_at",)


# ------------------------------------------------------------
# OPTIONAL: make Django's built-in User admin more helpful
# (Only if you want to see related profiles quickly)
# ------------------------------------------------------------
# If you already customized User admin elsewhere, skip this part.
#
# admin.site.unregister(User)
# @admin.register(User)
# class UserAdminCustom(admin.ModelAdmin):
#     list_display = ("username", "first_name", "last_name", "is_active", "is_staff", "is_superuser")
#     search_fields = ("username", "first_name", "last_name", "email")
#     list_filter = ("is_active", "is_staff", "is_superuser")
