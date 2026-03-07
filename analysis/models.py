from django.db import models
from django.contrib.auth.models import User

class Institution(models.Model):
    name = models.CharField(max_length=255)
    address = models.TextField()
    school_year = models.CharField(max_length=9)  # e.g., "2025-2026"
    
    def __str__(self):
        return self.name
    
class InstitutionAdmin(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # Ensure this field exists
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)  # Add an active status for the admin

    def __str__(self):
        return f"{self.user.username} - {self.institution.name}" 

class InstitutionDetails(models.Model):
    institution = models.OneToOneField(Institution, on_delete=models.CASCADE)
    grading_periods = models.JSONField(default=list)  # e.g., ["1st Grading", "2nd Grading"]
    
    def __str__(self):
        return f"Details for {self.institution.name}"
    
class SchoolYear(models.Model):
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE)
    year = models.CharField(max_length=9)  # Example: "2025-2026"
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    is_active = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["institution", "year"], name="uniq_sy_per_institution")
        ]

    def __str__(self):
        return f"{self.year} - {self.institution.name}"

class CustomUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.user.username
    
class GradingPeriod(models.Model):
    GRADING_CHOICES = [
        ('1st Grading', 'First'),
        ('2nd Grading', 'Second'),
        ('3rd Grading', 'Third'),
        ('4th Grading', 'Fourth'),
    ]

    institution = models.ForeignKey('Institution', on_delete=models.CASCADE, related_name='grading_periods')
    # Add this field to track which year the period belongs to
    school_year = models.CharField(max_length=20, null=True, blank=True) 
    period = models.CharField(max_length=15, choices=GRADING_CHOICES, null=True)
    start_date = models.DateField()
    end_date = models.DateField()

    def __str__(self):
        # Improved string representation to include the year
        return f"{self.period} ({self.school_year}) - {self.institution.name}"

class Grade(models.Model):
    institution = models.ForeignKey(
        Institution,
        on_delete=models.CASCADE,
        related_name="grades",
        null=True, 
        blank=True,  
    )
    grading_period = models.ForeignKey(GradingPeriod, on_delete=models.CASCADE, related_name='grades',  null=True,
    blank=True)

    school_year = models.ForeignKey(SchoolYear, on_delete=models.CASCADE, related_name="grades", null=True, blank=True)

    name = models.CharField(max_length=50)
    number = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["number", "name"]
        constraints = [
            models.UniqueConstraint(fields=["institution", "name"], name="uniq_grade_name_per_institution"),
            models.UniqueConstraint(fields=["institution", "number"], name="uniq_grade_number_per_institution"),
        ]

    def __str__(self):
        return f"{self.name} ({self.institution.name if self.institution else 'No Institution'})"

class Section(models.Model):
    # e.g. "Lotus", "Sampaguita"
    name = models.CharField(max_length=50)
    grade = models.ForeignKey(
        Grade,
        on_delete=models.CASCADE,
        related_name="sections",
    )

    class Meta:
        unique_together = ("name", "grade")
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.grade.name})"


# ----------------------------------------------------------------------
# SUBJECT (PER GRADE)
# ----------------------------------------------------------------------
class Subject(models.Model):
    grade = models.ForeignKey(
        Grade,
        on_delete=models.CASCADE,
        related_name="subjects",
    )
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    class Meta:
        unique_together = ("grade", "name")
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} - {self.grade.name}"

class Student(models.Model):
    institution = models.ForeignKey(
        Institution,
        on_delete=models.CASCADE,
        related_name="students",
        null=True,   # ✅ TEMPORARY for migration (make False later)
        blank=True,
    )

    school_year = models.CharField(
        max_length=9,
        blank=True,
        default="",
        help_text="Snapshot of the institution school year at time of saving (e.g., 2025-2026).",
    )

    grade = models.ForeignKey(
        Grade,
        on_delete=models.CASCADE,
        related_name="students",
        null=True,
        blank=True,
        help_text="Convenience link; usually same as section.grade.",
    )

    section = models.ForeignKey(
        Section,
        on_delete=models.CASCADE,
        related_name="students",
    )

    full_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255, blank=True, default="")
    middle_initial = models.CharField(max_length=10, blank=True, default="")

    # ✅ NEW: gender field (this is what your JS grouping needs)
    GENDER_CHOICES = (
        ("Male", "Male"),
        ("Female", "Female"),
    )
    gender = models.CharField(
        max_length=10,
        choices=GENDER_CHOICES,
        blank=True,
        null=True,
        default=None,
    )

    class Meta:
        ordering = ["last_name", "full_name"]
        # ✅ optional but recommended: prevent duplicates inside an institution + section
        constraints = [
            models.UniqueConstraint(
                fields=["institution", "section", "full_name", "last_name", "middle_initial"],
                name="uniq_student_name_per_institution_section",
            )
        ]

    def save(self, *args, **kwargs):
        # infer last name if empty
        if self.full_name and not self.last_name:
            parts = self.full_name.strip().split()
            if parts:
                self.last_name = parts[-1]

        # sync grade
        if self.section and self.grade is None:
            self.grade = self.section.grade

        # ✅ set institution + school year from grade/section
        if self.section and self.section.grade and self.section.grade.institution:
            self.institution = self.section.grade.institution
            self.school_year = self.section.grade.institution.school_year

        super().save(*args, **kwargs)

    @property
    def formatted_name(self):
        bits = []
        if self.last_name:
            bits.append(self.last_name + ",")
        bits.append(self.full_name)
        if self.middle_initial:
            bits.append(self.middle_initial + ".")
        return " ".join(bits).strip()

    def __str__(self):
        return self.formatted_name

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    institution = models.ForeignKey(
        Institution,
        on_delete=models.CASCADE,
        related_name="teachers",
        null=True,  
        blank=True,
    )
    school_year = models.CharField(max_length=9, blank=True, default="")

    grade = models.ForeignKey(Grade, on_delete=models.SET_NULL, null=True, blank=True)
    section = models.ForeignKey(Section, on_delete=models.SET_NULL, null=True, blank=True)
    subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True)

    def save(self, *args, **kwargs):
        if self.institution and not self.school_year:
            self.school_year = self.institution.school_year
        super().save(*args, **kwargs)

class TeacherClassAssignment(models.Model):
    teacher = models.ForeignKey(
        "Teacher",
        on_delete=models.CASCADE,
        related_name="class_assignments",
    )
    grade = models.ForeignKey("Grade", on_delete=models.CASCADE)
    section = models.ForeignKey("Section", on_delete=models.CASCADE)
    subject = models.ForeignKey("Subject", on_delete=models.CASCADE)

    # keep it consistent with your institution/school_year filtering style
    institution = models.ForeignKey("Institution", on_delete=models.CASCADE, null=True, blank=True)
    school_year = models.CharField(max_length=20, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["teacher", "grade", "section", "subject", "institution", "school_year"],
                name="uniq_teacher_grade_section_subject_per_sy",
            )
        ]

    def __str__(self):
        return f"{self.teacher} - {self.grade} / {self.section} / {self.subject}"

# ----------------------------------------------------------------------
# ANSWER KEY (PER GRADE / SECTION / SUBJECT)
# ----------------------------------------------------------------------

class AnswerKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="answer_keys")

    institution = models.ForeignKey(
        "Institution",
        on_delete=models.CASCADE,
        related_name="answer_keys",
        null=True,
        blank=True,
    )
    school_year = models.CharField(max_length=9, blank=True, default="", db_index=True)
    
    # ✅ NEW: Assessment Name (e.g., Quiz 1, Midterm)
    quiz_name = models.CharField(max_length=255, default="Default Quiz")

    # ✅ grading period per upload
    grading_period = models.ForeignKey(
        "GradingPeriod",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="answer_keys",
    )

    grade = models.ForeignKey("Grade", on_delete=models.CASCADE, related_name="answer_keys")
    section = models.ForeignKey("Section", on_delete=models.CASCADE, related_name="answer_keys")
    subject = models.ForeignKey("Subject", on_delete=models.CASCADE, related_name="answer_keys")

    file = models.FileField(upload_to="answer_keys/")
    test_sheet = models.FileField(upload_to="answer_key_sheets/", blank=True, null=True)

    item_points = models.JSONField(default=dict, blank=True)
    total_points = models.FloatField(default=0.0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    total_items = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-uploaded_at"]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "user", "institution", "school_year",
                    "grading_period",
                    "grade", "section", "subject",
                    "quiz_name" 
                ],
                name="uniq_answerkey_per_user_inst_year_gp_class_quiz",
                violation_error_message="An answer key with this quiz name already exists for this period."
            )
        ]

    def save(self, *args, **kwargs):
        if self.institution and not self.school_year:
            self.school_year = self.institution.school_year
        super().save(*args, **kwargs)

    def __str__(self):
        gp = self.grading_period.period if self.grading_period else "-"
        # Updated to include quiz_name for better identification in admin
        return f"{self.quiz_name} | {gp} | {self.grade.name} - {self.subject.name}"
    
    def clean(self):
        """Add validation to ensure school_year is present if institution is."""
        from django.core.exceptions import ValidationError
        if self.institution and not self.institution.school_year:
            raise ValidationError("The institution must have an active school year set.")


# ----------------------------------------------------------------------
# SCAN RESULT – EVERY SCANNED BUBBLE SHEET
# ----------------------------------------------------------------------
class ScanResult(models.Model):
    answer_key = models.ForeignKey("AnswerKey", on_delete=models.CASCADE, related_name="scan_results")

    institution = models.ForeignKey(
        "Institution",
        on_delete=models.CASCADE,
        related_name="scan_results",
        null=True,
        blank=True,
    )
    school_year = models.CharField(max_length=9, null=True, blank=True)

    # ✅ NEW: grading per scan
    grading_period = models.ForeignKey(
        "GradingPeriod",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="scan_results",
    )

    grade = models.ForeignKey("Grade", on_delete=models.CASCADE, related_name="scan_results")
    section = models.ForeignKey("Section", on_delete=models.CASCADE, related_name="scan_results")
    subject = models.ForeignKey("Subject", on_delete=models.CASCADE, related_name="scan_results")

    student_name = models.CharField(max_length=255, blank=True)
    sheet_image = models.ImageField(upload_to="scan_sheets/", blank=True, null=True)
    answers = models.JSONField(default=list, blank=True)

    score = models.FloatField(default=0.0)
    max_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    student = models.ForeignKey("Student", on_delete=models.CASCADE, related_name="scans", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["institution", "school_year", "grading_period", "created_at"]),
        ]
        constraints = [
            # ✅ prevent duplicate scan of same student in SAME grading
            models.UniqueConstraint(
                fields=["answer_key", "grading_period", "student_name"],
                name="uniq_scan_per_student_per_gp",
            )
        ]
        
    def save(self, *args, **kwargs):
        if self.answer_key:
            # ✅ always copy institution, school_year, grading_period from answer key
            self.institution = self.answer_key.institution
            self.school_year = self.answer_key.school_year
            self.grading_period = self.answer_key.grading_period

            # ✅ ensure grade/section/subject match answer key too
            self.grade = self.answer_key.grade
            self.section = self.answer_key.section
            self.subject = self.answer_key.subject

        super().save(*args, **kwargs)

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_users",
    )

    def __str__(self):
        return f"{self.user.username} (created_by={self.created_by})"
    
class ClassAssignment(models.Model):
    teacher = models.ForeignKey(User, on_delete=models.CASCADE, related_name='class_assignments')
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name='class_assignments')
    grade = models.ForeignKey("Grade", on_delete=models.CASCADE, related_name="class_assignments")
    section = models.ForeignKey("Section", on_delete=models.CASCADE, related_name="class_assignments")
    subject = models.ForeignKey("Subject", on_delete=models.CASCADE, related_name="class_assignments")
    grading_period = models.ForeignKey(GradingPeriod, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.teacher.username} - {self.grade.name} {self.section.name} {self.subject.name} - {self.grading_period.name}"

# models.py
class AssessmentRecord(models.Model):
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE)
    school_year = models.CharField(max_length=9)  # "2025-2026"

    grade = models.ForeignKey(Grade, on_delete=models.SET_NULL, null=True, blank=True)
    section = models.ForeignKey(Section, on_delete=models.SET_NULL, null=True, blank=True)
    subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True)

    teacher = models.ForeignKey("Teacher", on_delete=models.SET_NULL, null=True, blank=True)
    student = models.ForeignKey("Student", on_delete=models.SET_NULL, null=True, blank=True)

    score = models.IntegerField(default=0)
    max_score = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    # optional: store test metadata
    assessment_name = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        return f"{self.institution} {self.school_year} {self.student} {self.score}/{self.max_score}"
