window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function ()
{
	// Check for click events on the navbar burger icon

	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	bulmaSlider.attach();

	const videoSwitchButtons = document.querySelectorAll('.video-switch-btn');
	const videoContainers = new Map();

	// 预加载所有视频
	function preloadVideos()
	{
		const videoGroups = new Map();

		// 按目标视频ID分组
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

		// 为每个组创建多个video元素
		videoGroups.forEach((videos, targetVideoId) =>
		{
			const originalVideo = document.getElementById(targetVideoId);
			const container = originalVideo.parentNode;

			// 创建一个video容器来包装所有video元素
			const videoWrapper = document.createElement('div');
			videoWrapper.style.position = 'relative';
			videoWrapper.style.display = 'flex';
			videoWrapper.style.justifyContent = 'center';
			videoWrapper.style.alignItems = 'center';
			videoWrapper.style.flexShrink = '0';

			// 将原始video包装在容器中
			container.insertBefore(videoWrapper, originalVideo);
			videoWrapper.appendChild(originalVideo);

			const videoElements = new Map();

			videos.forEach((videoInfo, index) =>
			{
				let videoElement;

				if (index === 0)
				{
					// 使用原始video元素
					videoElement = originalVideo;
				} else
				{
					// 创建新的video元素
					videoElement = originalVideo.cloneNode(true);
					videoElement.id = `${targetVideoId}_${index}`;
					videoElement.style.position = 'absolute';
					videoElement.style.top = '0';
					videoElement.style.left = '0';
					videoElement.style.display = 'none';
					// 将新video添加到wrapper中，而不是container
					videoWrapper.appendChild(videoElement);
				}

				// 设置视频源
				const source = videoElement.querySelector('source');
				source.src = videoInfo.src;
				videoElement.load();

				// 同步视频播放
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

	videoSwitchButtons.forEach(button =>
	{
		button.addEventListener('click', function ()
		{
			const videoSrc = this.getAttribute('data-video');
			const targetVideoId = this.getAttribute('data-target');
			const container = videoContainers.get(targetVideoId);

			if (container && container.videos.has(videoSrc))
			{
				const currentVideo = container.currentVideo;
				const newVideo = container.videos.get(videoSrc);

				if (currentVideo !== newVideo)
				{
					// 同步播放状态
					newVideo.currentTime = currentVideo.currentTime;
					const isPlaying = !currentVideo.paused;

					// 切换显示 - 现在所有video都在同一个wrapper中
					currentVideo.style.display = 'none';
					newVideo.style.display = '';
					newVideo.style.position = currentVideo.style.position || 'relative';

					// 继续播放
					if (isPlaying)
					{
						newVideo.play();
					}

					// 更新当前视频引用
					container.currentVideo = newVideo;
					container.currentSrc = videoSrc;

					// 同步其他隐藏视频的时间
					container.videos.forEach((video, src) =>
					{
						if (video !== newVideo)
						{
							video.currentTime = newVideo.currentTime;
						}
					});
				}

				// 更新按钮状态
				const parentContainer = this.closest('.item-video');
				const allButtons = parentContainer.querySelectorAll('.video-switch-btn');
				allButtons.forEach(btn => btn.classList.remove('active'));
				this.classList.add('active');
			}
		});
	});


	// 初始化按钮状态
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

	// 延迟初始化以确保DOM完全加载
	setTimeout(() =>
	{
		preloadVideos();
		initializeButtonStates();
	}, 500);

})
