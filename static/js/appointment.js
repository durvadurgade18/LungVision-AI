const appointmentForm = document.querySelector("#appointment-form");

if (appointmentForm) {
    appointmentForm.addEventListener("submit", function (e) {

        const name = document.querySelector('input[name="patient_name"]').value;

        if (name.trim() === "") {
            e.preventDefault();
            alert("Please enter patient name.");
        }

    });
}