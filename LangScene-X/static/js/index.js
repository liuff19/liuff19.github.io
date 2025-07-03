window.HELP_IMPROVE_VIDEOJS = false;

let carousels;

const carouselOptions = {
	slidesToScroll: 1,
	slidesToShow: 1,
	loop: true,
	infinite: true,
	autoplay: true,
	autoplaySpeed: 5000,
};

const singlecarouselOptions = {
	slidesToScroll: 1,
	slidesToShow: 1,
	loop: false,
	infinite: false,
	autoplay: true,
	autoplaySpeed: 5000,
};

function initCarousels()
{
	if (carousels && carousels.forEach)
	{
		carousels.forEach(c =>
		{
			c.destroy();
			if (c.element && c.element.bulmaCarousel)
			{
				delete c.element.bulmaCarousel;
			}
		});
	}
	carousels = bulmaCarousel.attach('.carousel', carouselOptions);
}

$(document).ready(function ()
{
	// Check for click events on the navbar burger icon

	// Initialize all div with carousel class
	initCarousels();
	bulmaSlider.attach();
	document.querySelectorAll('.navbar-item').forEach(item =>
	{
		item.addEventListener('click', () =>
		{
			setTimeout(initCarousels, 0);
		});
	});

	const videoSwitchButtons = document.querySelectorAll('.video-switch-btn');
	const videoContainers = new Map();

	function preloadVideos()
	{
		const videoGroups = new Map();

		videoSwitchButtons.forEach(button =>
		{
			const targetVideoId = button.getAttribute('data-target');
			const videoSrc = button.getAttribute('data-video');

			if (!videoGroups.has(targetVideoId))
			{
				videoGroups.set(targetVideoId, []);
			}
			videoGroups.get(targetVideoId).push({
				src: videoSrc,
				button: button
			});
		});

		videoGroups.forEach((videos, targetVideoId) =>
		{
			const originalVideo = document.getElementById(targetVideoId);
			const container = originalVideo.parentNode;

			const videoWrapper = document.createElement('div');
			videoWrapper.style.position = 'relative';
			videoWrapper.style.display = 'flex';
			videoWrapper.style.justifyContent = 'center';
			videoWrapper.style.alignItems = 'center';
			videoWrapper.style.flexShrink = '0';

			container.insertBefore(videoWrapper, originalVideo);
			videoWrapper.appendChild(originalVideo);

			const videoElements = new Map();

			videos.forEach((videoInfo, index) =>
			{
				let videoElement;

				if (index === 0)
				{
					videoElement = originalVideo;
				} else
				{
					videoElement = originalVideo.cloneNode(true);
					videoElement.id = `${targetVideoId}_${index}`;
					videoElement.style.position = 'absolute';
					videoElement.style.top = '0';
					videoElement.style.left = '0';
					videoElement.style.display = 'none';
					videoWrapper.appendChild(videoElement);
				}

				const source = videoElement.querySelector('source');
				source.src = videoInfo.src;
				videoElement.load();

				videoElement.addEventListener('loadeddata', () =>
				{
					if (originalVideo.readyState >= 2)
					{
						videoElement.currentTime = originalVideo.currentTime;
						if (!originalVideo.paused)
						{
							videoElement.play();
						}
					}
				});

				videoElements.set(videoInfo.src, videoElement);
			});

			videoContainers.set(targetVideoId, {
				videos: videoElements,
				currentSrc: originalVideo.querySelector('source').src,
				currentVideo: originalVideo,
				wrapper: videoWrapper
			});
		});
	}

	document.addEventListener('click', function (event)
	{
		if (event.target.classList.contains('video-switch-btn'))
		{
			const button = event.target;
			const videoSrc = button.getAttribute('data-video');
			const targetVideoId = button.getAttribute('data-target');
			const container = videoContainers.get(targetVideoId);

			if (container && container.videos.has(videoSrc))
			{
				const currentVideo = container.currentVideo;
				const newVideo = container.videos.get(videoSrc);

				if (currentVideo !== newVideo)
				{
					newVideo.currentTime = currentVideo.currentTime;
					const isPlaying = !currentVideo.paused;

					currentVideo.style.display = 'none';
					newVideo.style.display = '';
					newVideo.style.position = currentVideo.style.position || 'relative';

					if (isPlaying)
					{
						newVideo.play();
					}

					container.currentVideo = newVideo;
					container.currentSrc = videoSrc;

					container.videos.forEach((video, src) =>
					{
						if (video !== newVideo)
						{
							video.currentTime = newVideo.currentTime;
						}
					});
				}

				const parentContainer = button.closest('.item-video');
				const allButtons = parentContainer.querySelectorAll('.video-switch-btn');
				allButtons.forEach(btn => btn.classList.remove('active'));
				button.classList.add('active');
			}
		}
	});


	// videoSwitchButtons.forEach(button =>
	// {
	// 	button.addEventListener('click', function ()
	// 	{
	// 		const videoSrc = this.getAttribute('data-video');
	// 		const targetVideoId = this.getAttribute('data-target');
	// 		const container = videoContainers.get(targetVideoId);

	// 		if (container && container.videos.has(videoSrc))
	// 		{
	// 			const currentVideo = container.currentVideo;
	// 			const newVideo = container.videos.get(videoSrc);

	// 			if (currentVideo !== newVideo)
	// 			{
	// 				newVideo.currentTime = currentVideo.currentTime;
	// 				const isPlaying = !currentVideo.paused;

	// 				currentVideo.style.display = 'none';
	// 				newVideo.style.display = '';
	// 				newVideo.style.position = currentVideo.style.position || 'relative';

	// 				if (isPlaying)
	// 				{
	// 					newVideo.play();
	// 				}

	// 				container.currentVideo = newVideo;
	// 				container.currentSrc = videoSrc;

	// 				container.videos.forEach((video, src) =>
	// 				{
	// 					if (video !== newVideo)
	// 					{
	// 						video.currentTime = newVideo.currentTime;
	// 					}
	// 				});
	// 			}

	// 			const parentContainer = this.closest('.item-video');
	// 			const allButtons = parentContainer.querySelectorAll('.video-switch-btn');
	// 			allButtons.forEach(btn => btn.classList.remove('active'));
	// 			this.classList.add('active');
	// 		}
	// 	});
	// });


	function initializeButtonStates()
	{
		videoSwitchButtons.forEach(button =>
		{
			const targetVideoId = button.getAttribute('data-target');
			const targetVideo = document.getElementById(targetVideoId);
			const buttonVideoSrc = button.getAttribute('data-video');
			const currentVideoSrc = targetVideo.querySelector('source').src;

			if (currentVideoSrc.includes(buttonVideoSrc.split('/').pop()))
			{
				button.classList.add('active');
			}
		});
	}

	setTimeout(() =>
	{
		preloadVideos();
		initializeButtonStates();
	}, 500);

})
