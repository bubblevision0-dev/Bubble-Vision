from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import Institution, InstitutionAdmin, Grade, Section, Subject, AnswerKey, Student, Teacher, SchoolYear, GradingPeriod, ClassAssignment

class UserAdminForm(forms.ModelForm):
    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)
    username = forms.CharField(max_length=150)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name']

    def save(self, commit=True):
        # ✅ No password logic here anymore
        user = super().save(commit=False)
        if commit:
            user.save()
        return user

# Institution Form (Create & Edit)
class InstitutionForm(forms.ModelForm):
    class Meta:
        model = Institution
        fields = ['name', 'address', 'school_year']

# Institution Admin Form (Create & Edit)
class InstitutionAdminForm(forms.ModelForm):
    class Meta:
        model = InstitutionAdmin
        fields = ['user', 'institution']
    
    def __init__(self, *args, **kwargs):
        institution = kwargs.get('institution', None)
        super().__init__(*args, **kwargs)
        if institution:
            self.fields['institution'].queryset = Institution.objects.filter(id=institution.id)

    def clean_user(self):
        user = self.cleaned_data.get('user')
        # Ensure the user is not already an admin for another institution
        if InstitutionAdmin.objects.filter(user=user).exists():
            raise forms.ValidationError("This user is already an admin for another institution.")
        return user

class SchoolYearForm(forms.ModelForm):
    class Meta:
        model = SchoolYear
        fields = ['year', 'start_date', 'end_date']
        widgets = {
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
        }
        
# Only Admin & Teacher
ROLE_CHOICES = [
    ('admin', 'Admin'),
    ('teacher', 'Teacher'),
]

class AddUserForm(forms.Form):
    first_name = forms.CharField(max_length=150, label="First Name")
    last_name  = forms.CharField(max_length=150, label="Last Name")
    username   = forms.CharField(max_length=150, label="Username")

    # optional assignments
    grade = forms.ModelChoiceField(queryset=Grade.objects.none(), required=False)
    section = forms.ModelChoiceField(queryset=Section.objects.none(), required=False)
    subject = forms.ModelChoiceField(queryset=Subject.objects.none(), required=False)

    def __init__(self, *args, **kwargs):
        institution = kwargs.pop("institution", None)
        super().__init__(*args, **kwargs)

        if institution:
            self.fields["grade"].queryset = Grade.objects.filter(institution=institution).order_by("number")
            self.fields["section"].queryset = Section.objects.filter(grade__institution=institution).order_by("name")
            self.fields["subject"].queryset = Subject.objects.filter(grade__institution=institution).order_by("name")

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Username already exists.")
        return username

    def save(self, institution):
        cd = self.cleaned_data
        first_name = cd["first_name"].strip()
        last_name  = cd["last_name"].strip()
        username   = cd["username"].strip()

        default_password = "@Teacher2025"

        user = User.objects.create_user(
            username=username,
            password=default_password,
            first_name=first_name,
            last_name=last_name,
        )

        Teacher.objects.create(
            user=user,
            institution=institution,
            school_year=institution.school_year,
            grade=cd.get("grade"),
            section=cd.get("section"),
            subject=cd.get("subject"),
        )

        return user
    
class EditUserForm(forms.Form):
    username = forms.CharField()

    grade = forms.ModelChoiceField(queryset=Grade.objects.none(), required=False)
    section = forms.ModelChoiceField(queryset=Section.objects.none(), required=False)
    subject = forms.ModelChoiceField(queryset=Subject.objects.none(), required=False)

    def __init__(self, *args, **kwargs):
        institution = kwargs.pop("institution", None)
        teacher = kwargs.pop("teacher", None)
        super().__init__(*args, **kwargs)

        if institution:
            self.fields["grade"].queryset = Grade.objects.filter(institution=institution).order_by("number")
            self.fields["section"].queryset = Section.objects.filter(grade__institution=institution).order_by("name")
            self.fields["subject"].queryset = Subject.objects.filter(grade__institution=institution).order_by("name")

        if teacher:
            self.initial["username"] = teacher.user.username
            self.initial["grade"] = teacher.grade_id
            self.initial["section"] = teacher.section_id
            self.initial["subject"] = teacher.subject_id

    def save(self, teacher, institution, grading_periods=None):
        cd = self.cleaned_data
        
        # 1. Update User basic info
        user = teacher.user
        user.first_name = cd["first_name"].strip()
        user.last_name = cd["last_name"].strip()
        user.username = cd["username"].strip()
        user.save()

        # 2. Clear current SY assignments to rebuild them
        TeacherClassAssignment.objects.filter(
            teacher=teacher, 
            school_year=institution.school_year
        ).delete()

        # 3. Save the updated dynamic rows
        grades = self.data.getlist('assign_grade[]')
        sections = self.data.getlist('assign_section[]')
        subjects = self.data.getlist('assign_subject[]')

        for i in range(len(grades)):
            if grades[i] and sections[i] and subjects[i]:
                period = grading_periods[i] if grading_periods and i < len(grading_periods) else ""
                
                TeacherClassAssignment.objects.create(
                    teacher=teacher,
                    grade_id=grades[i],
                    section_id=sections[i],
                    subject_id=subjects[i],
                    grading_period=period,
                    school_year=institution.school_year
                )
        
        return teacher

    
class GradeForm(forms.ModelForm):
    class Meta:
        model = Grade
        fields = ['grading_period','name']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Enter grade (e.g., Grade 10)'})
        }

    def clean_number(self):
        number = self.cleaned_data['number']
        if Grade.objects.filter(number=number).exists():
            raise forms.ValidationError("This grade number already exists.")
        return number
    
class SectionForm(forms.ModelForm):
    class Meta:
        model = Section
        fields = ['name']
        labels = {
            'name': 'Section Name'
        }
        
    def __init__(self, *args, **kwargs):
        self.grade = kwargs.pop('grade', None)
        super().__init__(*args, **kwargs)

    def clean_name(self):
        name = self.cleaned_data['name']
        if Section.objects.filter(name__iexact=name, grade=self.grade).exists():
            raise forms.ValidationError(f"Section '{name}' already exists for this grade.")
        return name
    
class SubjectForm(forms.ModelForm):
    class Meta:
        model = Subject
        fields = ['name']
        
class AnswerKeyForm(forms.ModelForm):
    class Meta:
        model = AnswerKey
        fields = ["grade", "section", "subject", "file", "test_sheet"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Limit sections and subjects dynamically based on grade
        if self.instance and self.instance.grade:
            self.fields['section'].queryset = Section.objects.filter(
                grade=self.instance.grade
            ).order_by("name")
            self.fields['subject'].queryset = Subject.objects.filter(
                grade=self.instance.grade
            ).order_by("name")
        else:
            self.fields['section'].queryset = Section.objects.none()
            self.fields['subject'].queryset = Subject.objects.none()

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ["full_name", "middle_initial", "last_name", "section"]

    def __init__(self, *args, **kwargs):
        institution = kwargs.pop("institution", None)
        super().__init__(*args, **kwargs)

        if institution:
            self.fields["section"].queryset = (
                Section.objects.select_related("grade")
                .filter(grade__institution=institution)
                .order_by("grade__number", "name")
            )

class UserCreationForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password']

    password = forms.CharField(widget=forms.PasswordInput())

class GradingPeriodForm(forms.ModelForm):
    class Meta:
        model = GradingPeriod
        fields = ['period', 'start_date', 'end_date']

    period = forms.ChoiceField(choices=GradingPeriod.GRADING_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}))