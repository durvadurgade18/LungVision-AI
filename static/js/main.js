//import "./modules/counter.js"
document.addEventListener("DOMContentLoaded", function () {

    console.log("LungVision AI loaded.");

});
const observer = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
        if(entry.isIntersecting){
            entry.target.classList.add("show");
        }
    });
});

document.querySelectorAll(".disease-card").forEach(card=>{
    observer.observe(card);
});
const learnBtn = document.getElementById("learnMoreBtn");
const modal = document.getElementById("videoModal");
const closeBtn = document.querySelector(".close-video");
const video = document.getElementById("heroVideo");

if (learnBtn && modal && closeBtn && video) {

    learnBtn.addEventListener("click", (e) => {
        e.preventDefault();
        modal.style.display = "flex";
        video.play();

if (video.requestFullscreen) {
    video.requestFullscreen();
} else if (video.webkitRequestFullscreen) {
    video.webkitRequestFullscreen(); // Safari
} else if (video.msRequestFullscreen) {
    video.msRequestFullscreen(); // Older Edge/IE
}
    });

    closeBtn.addEventListener("click", () => {
        modal.style.display = "none";
        video.pause();
        video.currentTime = 0;
    });

    window.addEventListener("click", (e) => {
        if (e.target === modal) {
            modal.style.display = "none";
            video.pause();
            video.currentTime = 0;
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            modal.style.display = "none";
            video.pause();
            video.currentTime = 0;
        }
    });

}

const watchDemoBtn = document.getElementById("watchDemoBtn");
const videoModal = document.getElementById("videoModal");
const closeVideo = document.querySelector(".close-video");
const demoVideo = document.getElementById("demoVideo");

if (watchDemoBtn) {

    watchDemoBtn.addEventListener("click", function(e){

        e.preventDefault();

        videoModal.style.display = "flex";

        demoVideo.play();

    });

}

if (closeVideo) {

    closeVideo.addEventListener("click", function(){

        videoModal.style.display = "none";

        demoVideo.pause();

        demoVideo.currentTime = 0;

    });

}

window.addEventListener("click", function(e){

    if(e.target == videoModal){

        videoModal.style.display = "none";

        demoVideo.pause();

        demoVideo.currentTime = 0;

    }

});