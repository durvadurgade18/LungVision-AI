// ===============================
// Password Show / Hide
// ===============================

const password = document.getElementById("password");
const togglePassword = document.querySelector(".togglePassword");

if (togglePassword && password) {
    togglePassword.addEventListener("click", () => {

        if (password.type === "password") {

            password.type = "text";

            togglePassword.classList.remove("fa-eye");
            togglePassword.classList.add("fa-eye-slash");

        } else {

            password.type = "password";

            togglePassword.classList.remove("fa-eye-slash");
            togglePassword.classList.add("fa-eye");

        }

    });
}

// ===============================
// Login Form
// ===============================

const loginForm = document.getElementById("loginForm");

if (loginForm) {

    loginForm.addEventListener("submit", function (e) {

        const email = document.querySelector("input[type='email']").value.trim();
        const pass = password.value.trim();

        if (email === "" || pass === "") {

            e.preventDefault();

            alert("Please enter your email and password.");

            return;
        }

        // Let the browser submit the form to Flask
    });

}

// ===============================
// Fade In Animation
// ===============================

window.addEventListener("load", () => {

    const card = document.querySelector(".login-card");
    const hero = document.querySelector(".hero-image");

    if (card) {
        card.animate(
            [
                { opacity: 0, transform: "translateX(-60px)" },
                { opacity: 1, transform: "translateX(0)" }
            ],
            {
                duration: 900,
                easing: "ease"
            }
        );
    }

    if (hero) {
        hero.animate(
            [
                { opacity: 0, transform: "translateX(60px)" },
                { opacity: 1, transform: "translateX(0)" }
            ],
            {
                duration: 900,
                easing: "ease"
            }
        );
    }

});