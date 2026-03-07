document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggle-btn");
  const sidebar = document.getElementById("sidebar");

  toggleBtn.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll(".toggle-password");

  toggles.forEach(toggle => {
    toggle.addEventListener("click", () => {
      const fieldId = toggle.getAttribute("data-target");
      const passwordField = document.getElementById(fieldId);
      if (passwordField.type === "password") {
        passwordField.type = "text";
        toggle.classList.remove("fa-eye");
        toggle.classList.add("fa-eye-slash");
      } else {
        passwordField.type = "password";
        toggle.classList.remove("fa-eye-slash");
        toggle.classList.add("fa-eye");
      }
    });
  });
});


document.querySelectorAll('.toggle-password').forEach(icon => {
    icon.addEventListener('click', function() {
        const target = document.getElementById(this.dataset.target);
        if (target.type === 'password') {
            target.type = 'text';
            this.classList.add('fa-eye-slash');
        } else {
            target.type = 'password';
            this.classList.remove('fa-eye-slash');
        }
    });
});

