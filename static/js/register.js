const registerForm = document.querySelector("form");

if (registerForm) {

    registerForm.addEventListener("submit", function (e) {

        const passwords = document.querySelectorAll('input[type="password"]');

        if (passwords[0].value !== passwords[1].value) {

            e.preventDefault();

            alert("Passwords do not match!");

            return;
        }

        // Allow the form to submit to Flask
    });

}