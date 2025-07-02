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
				if (source)
				{
					source.src = videoInfo.src;
					videoElement.load();

					// 改进的加载事件监听
					const handleLoadedData = () =>
					{
						try
						{
							if (originalVideo.readyState >= 2)
							{
								videoElement.currentTime = originalVideo.currentTime;
								if (!originalVideo.paused)
								{
									videoElement.play().catch(e => console.warn('Auto-play failed:', e));
								}
							}
						} catch (error)
						{
							console.warn('Error syncing video:', error);
						}
					};

					videoElement.addEventListener('loadeddata', handleLoadedData);
					
					// 添加错误处理
					videoElement.addEventListener('error', (e) =>
					{
						console.error('Video load error:', videoInfo.src, e);
					});
				} else
				{
					console.error('No source element found in video:', videoElement);
				}

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

			console.log('Button clicked:', { videoSrc, targetVideoId, containerExists: !!container });

			if (!container)
			{
				console.error('Container not found for:', targetVideoId);
				return;
			}

			if (!container.videos.has(videoSrc))
			{
				console.error('Video not found:', videoSrc, 'Available videos:', Array.from(container.videos.keys()));
				return;
			}

			const currentVideo = container.currentVideo;
			const newVideo = container.videos.get(videoSrc);

			if (currentVideo !== newVideo)
			{
				// 确保新视频已加载
				if (newVideo.readyState < 2)
				{
					console.log('Video not ready, waiting...');
					newVideo.addEventListener('loadeddata', () =>
					{
						performVideoSwitch(currentVideo, newVideo, container, videoSrc);
					}, { once: true });
					return;
				}

				performVideoSwitch(currentVideo, newVideo, container, videoSrc);
			}

			// 更新按钮状态
			const parentContainer = this.closest('.item-video');
			const allButtons = parentContainer.querySelectorAll('.video-switch-btn');
			allButtons.forEach(btn => btn.classList.remove('active'));
			this.classList.add('active');
		});
	});

	// 提取视频切换逻辑到单独函数
	function performVideoSwitch(currentVideo, newVideo, container, videoSrc)
	{
		try
		{
			const currentTime = currentVideo.currentTime;
			const isPlaying = !currentVideo.paused;

			// 设置新视频时间
			newVideo.currentTime = currentTime;

			// 切换显示
			currentVideo.style.display = 'none';
			newVideo.style.display = '';
			newVideo.style.position = currentVideo.style.position || 'relative';

			// 继续播放
			if (isPlaying)
			{
				newVideo.play().catch(e => console.warn('Play failed:', e));
			}

			// 更新容器状态
			container.currentVideo = newVideo;
			container.currentSrc = videoSrc;

			// 同步其他视频时间
			container.videos.forEach((video, src) =>
			{
				if (video !== newVideo)
				{
					try
					{
						video.currentTime = currentTime;
					} catch (e)
					{
						console.warn('Failed to sync video time:', e);
					}
				}
			});

			console.log('Video switched successfully to:', videoSrc);
		} catch (error)
		{
			console.error('Error during video switch:', error);
		}
	}


	function initializeButtonStates()
	{
		videoSwitchButtons.forEach(button =>
		{
			try
			{
				const targetVideoId = button.getAttribute('data-target');
				const targetVideo = document.getElementById(targetVideoId);
				const buttonVideoSrc = button.getAttribute('data-video');
				
				if (!targetVideo)
				{
					console.warn('Target video not found:', targetVideoId);
					return;
				}
				
				const sourceElement = targetVideo.querySelector('source');
				if (!sourceElement)
				{
					console.warn('Source element not found for video:', targetVideoId);
					return;
				}
				
				const currentVideoSrc = sourceElement.src;

				if (currentVideoSrc.includes(buttonVideoSrc.split('/').pop()))
				{
					button.classList.add('active');
					console.log('Button activated:', button.textContent);
				}
			} catch (error)
			{
				console.error('Error initializing button state:', error);
			}
		});
	}

	setTimeout(() =>
	{
		preloadVideos();
		initializeButtonStates();
	}, 500);

})
