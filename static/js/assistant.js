const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");

/* ===============================
   ADD MESSAGE
================================ */

function addMessage(message, sender) {

    const div = document.createElement("div");

    div.className =
        sender === "user"
            ? "user-message"
            : "bot-message";

    div.innerHTML = marked.parse(message);

    chatBox.appendChild(div);

    chatBox.scrollTop = chatBox.scrollHeight;
}

/* ===============================
   GET USER LOCATION
================================ */

async function getLocation() {

    return new Promise((resolve, reject) => {

        if (!navigator.geolocation) {
            reject("Geolocation not supported");
            return;
        }

        navigator.geolocation.getCurrentPosition(

            (position) => {

                resolve({

                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude

                });

            },

            (error) => {

                reject(error);

            }

        );

    });

}

/* ===============================
   SEND MESSAGE
================================ */

async function sendMessage() {

    const message = userInput.value.trim();

    if (message === "") return;

    addMessage(message, "user");

    userInput.value = "";

    const typing = document.createElement("div");

    typing.className = "bot-message";
    typing.id = "typing";
    typing.innerHTML = "🤖 Thinking...";

    chatBox.appendChild(typing);

    chatBox.scrollTop = chatBox.scrollHeight;

    let latitude = null;
    let longitude = null;

    if (
        message.toLowerCase().includes("near") ||
        message.toLowerCase().includes("hospital")
    ) {

        try {

            const location = await getLocation();

            latitude = location.latitude;
            longitude = location.longitude;

        } catch (error) {

            console.log(error);

        }

    }

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                message: message,
                latitude: latitude,
                longitude: longitude

            })

        });

        const data = await response.json();

        document.getElementById("typing").remove();

        addMessage(data.reply, "bot");

    }

    catch (error) {

        document.getElementById("typing").remove();

        addMessage(
            "❌ Unable to connect to the AI assistant.",
            "bot"
        );

        console.error(error);

    }

}

/* ===============================
   VOICE RECOGNITION
================================ */

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

if (SpeechRecognition) {

    const recognition = new SpeechRecognition();

    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;

    micBtn.addEventListener("click", () => {

        recognition.start();

    });

    
recognition.onstart = () => {

    micBtn.classList.add("listening");

    micBtn.innerHTML =
        '<i class="fa-solid fa-microphone-lines"></i>';

};

recognition.onend = () => {

    micBtn.classList.remove("listening");

    micBtn.innerHTML =
        '<i class="fa-solid fa-microphone"></i>';

};

    recognition.onresult = (event) => {

    const transcript = event.results[0][0].transcript;

    userInput.value = transcript;

    // Automatically send the message
    sendMessage();

};


    recognition.onerror = (event) => {

        console.log("Speech Recognition Error:", event.error);

    };

} else {

    console.log("Speech Recognition not supported.");

}

/* ===============================
   EVENTS
================================ */

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", function (e) {

    if (e.key === "Enter") {

        sendMessage();

    }

});