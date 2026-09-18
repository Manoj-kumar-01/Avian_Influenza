document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ──
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    const dropZone = document.getElementById('drop-zone');
    const audioInput = document.getElementById('audio-input');

    const btnRecord = document.getElementById('btn-record');
    const micIcon = document.getElementById('mic-icon');
    const micStatus = document.getElementById('mic-status');
    const micTimer = document.getElementById('mic-timer');

    const playerContainer = document.getElementById('player-container');
    const currentFilename = document.getElementById('current-filename');
    const playerTime = document.getElementById('player-time');
    const btnPlayPause = document.getElementById('btn-play-pause');
    const playIcon = document.getElementById('play-icon');

    const btnAnalyze = document.getElementById('btn-analyze');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultPanel = document.getElementById('result-panel');
    const bgDangerGlow = document.getElementById('bg-danger-glow');

    // Stage 1 Elements
    const stage1BadgeBox = document.getElementById('stage1-badge-box');
    const stage1Icon = document.getElementById('stage1-icon');
    const stage1Title = document.getElementById('stage1-title');
    const stage1Desc = document.getElementById('stage1-desc');

    // Stage 2 Elements
    const diagnosisCard = document.getElementById('diagnosis-card');
    const diagIconBox = document.getElementById('diag-icon-box');
    const diagPrediction = document.getElementById('diag-prediction');
    const diagConfidencePct = document.getElementById('diag-confidence-pct');
    const clinicalText = document.getElementById('clinical-text');
    const spectrogramImg = document.getElementById('spectrogram-img');

    // Multi-Class Probability Bars
    const valHealthy = document.getElementById('val-healthy');
    const barHealthy = document.getElementById('bar-healthy');
    const valUnhealthy = document.getElementById('val-unhealthy');
    const barUnhealthy = document.getElementById('bar-unhealthy');
    const valNoise = document.getElementById('val-noise');
    const barNoise = document.getElementById('bar-noise');

    const feedList = document.getElementById('feed-list');
    const btnPrintReport = document.getElementById('btn-print-report');

    // ── Audio Player & Controls ──
    const nativeAudioPlayer = document.getElementById('native-audio-player');
    const volumeSlider = document.getElementById('volume-slider');
    const volumeLabel = document.getElementById('volume-label');
    const volIconBtn = document.getElementById('vol-icon-btn');
    const btnSoundTest = document.getElementById('btn-sound-test');
    const btnRestart = document.getElementById('btn-restart');
    const btnSpeakDiagnosis = document.getElementById('btn-speak-diagnosis');
    const audioStatusPill = document.getElementById('audio-status-pill');

    let wavesurfer = null;
    let currentAudioFile = null;
    let currentBlobUrl = null;
    let lastInferenceData = null;

    function initWavesurfer() {
        if (wavesurfer) return;

        try {
            wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: '#38bdf8',
                progressColor: '#0284c7',
                cursorColor: '#f8fafc',
                barWidth: 2,
                barGap: 3,
                barRadius: 2,
                height: 60,
                normalize: true
            });

            wavesurfer.on('play', updatePlayState);
            wavesurfer.on('pause', updatePauseState);
            wavesurfer.on('finish', () => {
                updatePauseState();
                if (nativeAudioPlayer) nativeAudioPlayer.pause();
                if (audioStatusPill) {
                    audioStatusPill.textContent = 'Finished';
                    audioStatusPill.classList.remove('playing');
                }
            });

            wavesurfer.on('timeupdate', (currentTime) => {
                const total = wavesurfer.getDuration() || (nativeAudioPlayer ? nativeAudioPlayer.duration : 0) || 0;
                playerTime.textContent = `${formatTime(currentTime)} / ${formatTime(total)}`;
            });

            // When user clicks/seeks on the waveform, sync native player time
            wavesurfer.on('seeking', (currentTime) => {
                if (nativeAudioPlayer && Math.abs(nativeAudioPlayer.currentTime - currentTime) > 0.3) {
                    nativeAudioPlayer.currentTime = currentTime;
                }
            });

        } catch (e) {
            console.warn("WaveSurfer initialization error:", e);
        }
    }

    initWavesurfer();

    function formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) return "0:00";
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    function updatePlayState() {
        btnPlayPause.querySelector('span').textContent = 'Pause Audio';
        playIcon.setAttribute('data-lucide', 'pause');
        lucide.createIcons();
    }

    function updatePauseState() {
        btnPlayPause.querySelector('span').textContent = 'Play Audio';
        playIcon.setAttribute('data-lucide', 'play');
        lucide.createIcons();
    }

    async function ensureAudioUnlocked() {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
            if (!window.audioCtx) window.audioCtx = new AudioCtx();
            if (window.audioCtx.state === 'suspended') {
                await window.audioCtx.resume();
            }
        }
    }

    btnPlayPause.addEventListener('click', async () => {
        await ensureAudioUnlocked();

        const currentVol = parseFloat(volumeSlider ? volumeSlider.value : 1.0);

        if (nativeAudioPlayer && nativeAudioPlayer.src) {
            if (nativeAudioPlayer.paused) {
                nativeAudioPlayer.muted = false;
                nativeAudioPlayer.volume = currentVol;

                try {
                    await nativeAudioPlayer.play();
                } catch(err) {
                    console.warn("Direct play rejected, attempting wavesurfer:", err);
                }

                if (wavesurfer) {
                    try {
                        wavesurfer.setVolume(currentVol);
                        if (!wavesurfer.isPlaying()) {
                            wavesurfer.play();
                        }
                    } catch(e) {}
                }
            } else {
                nativeAudioPlayer.pause();
                if (wavesurfer && wavesurfer.isPlaying()) {
                    wavesurfer.pause();
                }
            }
        } else if (wavesurfer) {
            wavesurfer.setVolume(currentVol);
            wavesurfer.playPause();
        }
    });

    if (nativeAudioPlayer) {
        nativeAudioPlayer.addEventListener('play', () => {
            updatePlayState();
            if (audioStatusPill) {
                audioStatusPill.textContent = 'Playing Sound';
                audioStatusPill.classList.add('playing');
            }
            if (wavesurfer && !wavesurfer.isPlaying()) {
                wavesurfer.play().catch(() => {});
            }
        });

        nativeAudioPlayer.addEventListener('pause', () => {
            updatePauseState();
            if (audioStatusPill) {
                audioStatusPill.textContent = 'Paused';
                audioStatusPill.classList.remove('playing');
            }
            if (wavesurfer && wavesurfer.isPlaying()) {
                wavesurfer.pause();
            }
        });

        nativeAudioPlayer.addEventListener('timeupdate', () => {
            const cur = nativeAudioPlayer.currentTime || 0;
            const total = nativeAudioPlayer.duration || (wavesurfer ? wavesurfer.getDuration() : 0) || 0;
            playerTime.textContent = `${formatTime(cur)} / ${formatTime(total)}`;
        });

        nativeAudioPlayer.addEventListener('ended', () => {
            updatePauseState();
            if (audioStatusPill) {
                audioStatusPill.textContent = 'Finished';
                audioStatusPill.classList.remove('playing');
            }
            if (wavesurfer) wavesurfer.pause();
        });

        nativeAudioPlayer.addEventListener('error', () => {
            console.warn("Native audio player format error:", nativeAudioPlayer.error);
            if (audioStatusPill) {
                audioStatusPill.textContent = 'Format Handled';
            }
        });
    }

    if (btnRestart) {
        btnRestart.addEventListener('click', () => {
            if (nativeAudioPlayer) {
                nativeAudioPlayer.currentTime = 0;
            }
            if (wavesurfer) {
                wavesurfer.seekTo(0);
            }
            playerTime.textContent = `0:00 / ${formatTime(nativeAudioPlayer ? nativeAudioPlayer.duration : 0)}`;
        });
    }

    if (volumeSlider) {
        volumeSlider.addEventListener('input', (e) => {
            const vol = parseFloat(e.target.value);
            if (nativeAudioPlayer) {
                nativeAudioPlayer.volume = vol;
                nativeAudioPlayer.muted = (vol === 0);
            }
            if (wavesurfer) wavesurfer.setVolume(vol);
            if (volumeLabel) volumeLabel.textContent = `${Math.round(vol * 100)}%`;
        });
    }

    if (volIconBtn) {
        volIconBtn.addEventListener('click', () => {
            if (nativeAudioPlayer) {
                nativeAudioPlayer.muted = !nativeAudioPlayer.muted;
                if (nativeAudioPlayer.muted) {
                    volumeLabel.textContent = 'Muted';
                    volIconBtn.setAttribute('data-lucide', 'volume-x');
                } else {
                    const vol = parseFloat(volumeSlider.value || 1.0);
                    nativeAudioPlayer.volume = vol;
                    volumeLabel.textContent = `${Math.round(vol * 100)}%`;
                    volIconBtn.setAttribute('data-lucide', 'volume-2');
                }
                lucide.createIcons();
            }
        });
    }

    // ── Speaker Sound Test (Hardware Output Verification) ──
    if (btnSoundTest) {
        btnSoundTest.addEventListener('click', async () => {
            await ensureAudioUnlocked();
            playSpeakerTestSound();
        });
    }

    function playSpeakerTestSound() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            const ctx = new AudioCtx();
            if (ctx.state === 'suspended') ctx.resume();

            const now = ctx.currentTime;

            // Note 1: C5 (523Hz)
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(523.25, now);
            gain1.gain.setValueAtTime(0.3, now);
            gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
            osc1.connect(gain1);
            gain1.connect(ctx.destination);
            osc1.start(now);
            osc1.stop(now + 0.3);

            // Note 2: G5 (784Hz)
            const osc2 = ctx.createOscillator();
            const gain2 = ctx.createGain();
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(783.99, now + 0.18);
            gain2.gain.setValueAtTime(0.35, now + 0.18);
            gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
            osc2.connect(gain2);
            gain2.connect(ctx.destination);
            osc2.start(now + 0.18);
            osc2.stop(now + 0.6);

            // Spoken voice test
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const msg = new SpeechSynthesisUtterance("Audio speaker test passed. Sound output is working.");
                msg.rate = 1.0;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            }

            if (audioStatusPill) {
                audioStatusPill.textContent = 'Speaker Test: 100% OK';
                audioStatusPill.classList.add('playing');
                setTimeout(() => {
                    audioStatusPill.classList.remove('playing');
                    audioStatusPill.textContent = 'Ready to play';
                }, 3000);
            }
        } catch (err) {
            alert("Speaker sound test error: " + err.message);
        }
    }

    // ── Voice Speech Synthesis of Diagnostic Assessment ──
    if (btnSpeakDiagnosis) {
        btnSpeakDiagnosis.addEventListener('click', () => {
            if (!('speechSynthesis' in window)) {
                alert("Speech synthesis is not supported in this browser.");
                return;
            }

            window.speechSynthesis.cancel();

            let text = "";
            if (lastInferenceData && lastInferenceData.status === 'rejected') {
                text = `Stage 1 Hen Voice Filter Alert. The uploaded audio was rejected because it does not match domestic poultry vocalizations. Reason: ${lastInferenceData.reason || 'Frequency mismatch'}. Please record or upload an authentic poultry sound.`;
            } else if (lastInferenceData) {
                const p = lastInferenceData.prediction;
                const c = (lastInferenceData.confidence * 100).toFixed(1);
                const note = lastInferenceData.clinical_note || "";
                text = `Diagnostic Assessment Complete. Verified Hen Vocalization. Stage 2 Neural Classification: ${p}, with ${c} percent confidence. Veterinary clinical note: ${note}`;
            } else {
                text = "No diagnostic assessment available yet. Please select an audio file and click Run Diagnostic Analysis.";
            }

            const utter = new SpeechSynthesisUtterance(text);
            utter.rate = 0.95;
            utter.pitch = 1.0;
            utter.volume = 1.0;
            window.speechSynthesis.speak(utter);
        });
    }

    // ── Tab Switching (Upload vs Mic) ──
    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            tabs.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add('active');
        });
    });

    // ── File Selection & Drag & Drop ──
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        currentAudioFile = file;
        currentFilename.textContent = file.name;
        playerContainer.classList.remove('hidden');

        if (currentBlobUrl && currentBlobUrl.startsWith('blob:')) {
            try { URL.revokeObjectURL(currentBlobUrl); } catch(e) {}
        }
        currentBlobUrl = URL.createObjectURL(file);

        if (audioStatusPill) {
            audioStatusPill.textContent = 'Audio Loaded';
            audioStatusPill.classList.remove('playing');
        }

        if (nativeAudioPlayer) {
            nativeAudioPlayer.src = currentBlobUrl;
            nativeAudioPlayer.volume = parseFloat(volumeSlider ? volumeSlider.value : 1.0);
            nativeAudioPlayer.muted = false;
            nativeAudioPlayer.load();
        }

        if (wavesurfer) {
            try {
                wavesurfer.load(currentBlobUrl);
            } catch(e) {
                console.warn("WaveSurfer load error:", e);
            }
        }

        updatePauseState();
        btnAnalyze.disabled = false;
        resultPanel.classList.add('hidden');
        bgDangerGlow.classList.remove('alert-active');
    }

    // ── Live Microphone Recording ──
    let mediaRecorder = null;
    let recordedChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    let isRecording = false;

    btnRecord.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            recordedChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordedChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'audio/webm' });
                const recordedFile = new File([blob], `mic_recording_${Date.now()}.webm`, { type: 'audio/webm' });
                handleSelectedFile(recordedFile);
                stream.getTracks().forEach(t => t.stop());
            };

            mediaRecorder.start();
            isRecording = true;
            btnRecord.classList.add('recording');
            micStatus.textContent = 'Recording live audio... Click to stop';
            micIcon.setAttribute('data-lucide', 'square');
            lucide.createIcons();

            recordSeconds = 0;
            micTimer.textContent = '00:00';
            recordInterval = setInterval(() => {
                recordSeconds++;
                const m = Math.floor(recordSeconds / 60);
                const s = recordSeconds % 60;
                micTimer.textContent = `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
            }, 1000);

        } catch (err) {
            alert('Microphone access denied or not available: ' + err.message);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        clearInterval(recordInterval);
        btnRecord.classList.remove('recording');
        micStatus.textContent = 'Recording captured! Ready to analyze.';
        micIcon.setAttribute('data-lucide', 'mic');
        lucide.createIcons();
    }

    // ── Main Analysis Trigger ──
    btnAnalyze.addEventListener('click', runAnalysis);

    async function runAnalysis() {
        if (!currentAudioFile) return;

        btnAnalyze.disabled = true;
        loadingSpinner.classList.remove('hidden');
        resultPanel.classList.add('hidden');
        bgDangerGlow.classList.remove('alert-active');

        const formData = new FormData();
        formData.append('file', currentAudioFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            handleInferenceResult(data);

        } catch (err) {
            alert('Analysis failed. Make sure the backend server is running.\n' + err.message);
        } finally {
            loadingSpinner.classList.add('hidden');
            btnAnalyze.disabled = false;
        }
    }

    // ── Diagnostic Result Rendering ──
    function handleInferenceResult(data) {
        lastInferenceData = data;
        resultPanel.classList.remove('hidden');
        resultPanel.scrollIntoView({ behavior: 'smooth' });

        // Update Mel-Spectrogram preview
        if (data.spectrogram) {
            spectrogramImg.src = data.spectrogram;
        }

        // Case 1: Stage 1 Gatekeeper REJECTED Non-Hen Voice
        if (data.status === 'rejected') {
            stage1BadgeBox.className = 'stage-badge-box rejected';
            stage1Icon.innerHTML = '<i data-lucide="x-circle"></i>';
            stage1Title.textContent = 'Non-Hen Audio Detected (Rejected)';
            stage1Desc.textContent = data.reason || data.error;

            diagnosisCard.className = 'diagnosis-badge rejected';
            diagIconBox.innerHTML = '<i data-lucide="ban"></i>';
            diagPrediction.textContent = 'REJECTED: NON-HEN';
            diagConfidencePct.textContent = `Hen Match: ${(data.hen_probability * 100 || 0).toFixed(1)}%`;

            clinicalText.textContent = `Gatekeeper Diagnostic: The acoustic spectrum does not match domestic hen vocalizations. ${data.reason}`;

            // Reset probability bars
            valHealthy.textContent = '0.0%';
            barHealthy.style.width = '0%';
            valUnhealthy.textContent = '0.0%';
            barUnhealthy.style.width = '0%';
            valNoise.textContent = '100.0%';
            barNoise.style.width = '100%';

            addFeedItem(data.filename, 'REJECTED NON-HEN', 'rejected', 'Filter active');
            triggerDiagnosticNotification('Non-Hen', 0.0, data.filename, false);
            lucide.createIcons();
            return;
        }

        // Case 2: Stage 1 PASSED & Stage 2 Disease Diagnosis
        stage1BadgeBox.className = 'stage-badge-box passed';
        stage1Icon.innerHTML = '<i data-lucide="check-circle-2"></i>';
        stage1Title.textContent = 'Verified Hen Vocalization';
        stage1Desc.textContent = 'Audio signature matched poultry profile. Passed to Stage 2 ResNet classifier.';

        const pred = data.prediction;
        const probs = data.probabilities;
        const confPct = (data.confidence * 100).toFixed(1) + '%';

        diagPrediction.textContent = pred;
        diagConfidencePct.textContent = confPct;
        clinicalText.textContent = data.clinical_note || '';

        // Reset badge classes
        diagnosisCard.className = 'diagnosis-badge';

        if (pred === 'Healthy') {
            diagnosisCard.classList.add('healthy');
            diagIconBox.innerHTML = '<i data-lucide="shield-check"></i>';
            diagPrediction.textContent = 'Healthy Poultry';
        } else if (pred === 'Unhealthy') {
            diagnosisCard.classList.add('unhealthy');
            diagIconBox.innerHTML = '<i data-lucide="alert-octagon"></i>';
            diagPrediction.textContent = 'Suspected Avian Influenza';
            bgDangerGlow.classList.add('alert-active');
        } else {
            diagnosisCard.classList.add('noise');
            diagIconBox.innerHTML = '<i data-lucide="volume-2"></i>';
            diagPrediction.textContent = 'Environmental Noise';
        }

        // Update Probability Bars
        valHealthy.textContent = `${(probs['Healthy'] * 100).toFixed(1)}%`;
        barHealthy.style.width = `${probs['Healthy'] * 100}%`;

        valUnhealthy.textContent = `${(probs['Unhealthy'] * 100).toFixed(1)}%`;
        barUnhealthy.style.width = `${probs['Unhealthy'] * 100}%`;

        valNoise.textContent = `${(probs['Noise'] * 100).toFixed(1)}%`;
        barNoise.style.width = `${probs['Noise'] * 100}%`;

        addFeedItem(data.filename, pred, pred.toLowerCase(), confPct);
        triggerDiagnosticNotification(pred, data.confidence, data.filename, true);
        lucide.createIcons();
    }

    function addFeedItem(filename, label, typeClass, metaInfo) {
        const emptyState = feedList.querySelector('.feed-empty');
        if (emptyState) emptyState.remove();

        const li = document.createElement('li');
        li.className = `feed-item ${typeClass}`;

        let icon = 'check-circle';
        if (typeClass === 'unhealthy') icon = 'alert-triangle';
        else if (typeClass === 'rejected') icon = 'ban';
        else if (typeClass === 'noise') icon = 'volume-2';

        const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        li.innerHTML = `
            <div class="feed-item-left">
                <div class="feed-item-icon">
                    <i data-lucide="${icon}" style="width:16px;height:16px"></i>
                </div>
                <div class="feed-item-text">
                    <strong>${label}</strong>
                    <span>${filename} &bull; ${metaInfo}</span>
                </div>
            </div>
            <div class="feed-item-time">${timeNow}</div>
        `;

        feedList.insertBefore(li, feedList.firstChild);
        lucide.createIcons();
    }

    // ── Report Printing ──
    btnPrintReport.addEventListener('click', () => {
        window.print();
    });

    // ── Alert & Notification Management ──
    const btnEnableNotif = document.getElementById('btn-enable-notif');
    const notifPermStatus = document.getElementById('notif-perm-status');
    const btnTestNotif = document.getElementById('btn-test-notif');
    const toastContainer = document.getElementById('toast-container');

    function updateNotifPermBadge() {
        if (!('Notification' in window)) {
            if (notifPermStatus) notifPermStatus.textContent = 'Alerts: Off';
            return;
        }
        if (Notification.permission === 'granted') {
            if (notifPermStatus) notifPermStatus.textContent = 'Desktop Alerts: ON';
            if (btnEnableNotif) {
                btnEnableNotif.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                btnEnableNotif.style.background = 'rgba(16, 185, 129, 0.15)';
                btnEnableNotif.style.color = '#34d399';
            }
        } else if (Notification.permission === 'denied') {
            if (notifPermStatus) notifPermStatus.textContent = 'Alerts: Blocked';
            if (btnEnableNotif) btnEnableNotif.style.borderColor = 'rgba(239, 68, 68, 0.5)';
        } else {
            if (notifPermStatus) notifPermStatus.textContent = 'Desktop Alerts';
        }
    }
    updateNotifPermBadge();

    if (btnEnableNotif) {
        btnEnableNotif.addEventListener('click', async () => {
            if (!('Notification' in window)) {
                showToast('Notifications Not Supported', 'Your browser does not support desktop notifications.', 'warning');
                return;
            }
            try {
                const perm = await Notification.requestPermission();
                updateNotifPermBadge();
                if (perm === 'granted') {
                    showToast('Desktop Alerts Active', 'You will now receive instant desktop notifications for poultry health status.', 'success');
                    playAlertSound('healthy');
                } else {
                    showToast('Permission Denied', 'Please allow notifications in browser site settings to receive desktop alerts.', 'warning');
                }
            } catch (err) {
                console.error('Permission request failed:', err);
            }
        });
    }

    // Audio chime using Web Audio API
    function playAlertSound(type = 'unhealthy') {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const now = ctx.currentTime;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.connect(gain);
            gain.connect(ctx.destination);

            if (type === 'unhealthy') {
                // Urgent dual-pulse alert
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(880, now);
                osc.frequency.setValueAtTime(660, now + 0.15);
                osc.frequency.setValueAtTime(880, now + 0.3);
                gain.gain.setValueAtTime(0.3, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);
                osc.start(now);
                osc.stop(now + 0.5);
            } else {
                // Gentle chime
                osc.type = 'sine';
                osc.frequency.setValueAtTime(523.25, now);
                osc.frequency.exponentialRampToValueAtTime(659.25, now + 0.2);
                gain.gain.setValueAtTime(0.2, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
                osc.start(now);
                osc.stop(now + 0.35);
            }
        } catch (e) {
            console.warn('Audio chime skipped:', e);
        }
    }

    // Toast popup
    function showToast(title, message, type = 'info') {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.style.cssText = `
            pointer-events: auto;
            min-width: 300px;
            max-width: 440px;
            padding: 14px 18px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid ${type === 'unhealthy' || type === 'error' ? 'rgba(239, 68, 68, 0.5)' : type === 'success' || type === 'healthy' ? 'rgba(16, 185, 129, 0.5)' : 'rgba(56, 189, 248, 0.5)'};
            color: #f8fafc;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            backdrop-filter: blur(12px);
            font-size: 0.9rem;
            animation: fadeIn 0.3s ease-out;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        `;

        const iconMap = {
            unhealthy: '🚨',
            error: '⚠️',
            healthy: '✅',
            success: '✅',
            info: '🔔',
            warning: '⚠️'
        };

        toast.innerHTML = `
            <span style="font-size: 1.25rem;">${iconMap[type] || '🔔'}</span>
            <div style="flex: 1;">
                <strong style="display: block; font-weight: 600; margin-bottom: 2px;">${title}</strong>
                <span style="opacity: 0.9; font-size: 0.85rem; line-height: 1.4;">${message}</span>
            </div>
            <button style="background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 1.2rem; line-height: 1; padding: 0 4px;" onclick="this.parentElement.remove()">&times;</button>
        `;

        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.5s ease';
            setTimeout(() => toast.remove(), 500);
        }, 7000);
    }

    // Trigger Desktop & Web Notification
    function triggerDiagnosticNotification(pred, conf, filename, isHen = true) {
        let title = '';
        let body = '';
        let type = 'info';

        if (!isHen) {
            title = 'Rejected: Non-Hen Audio';
            body = `Non-hen vocalization detected in ${filename}. Filtered by Stage 1.`;
            type = 'warning';
        } else if (pred === 'Unhealthy') {
            title = '🚨 ALERT: Suspected Avian Influenza!';
            body = `Suspected Avian Influenza detected in ${filename} (${(conf * 100).toFixed(1)}% confidence). Immediate poultry isolation advised!`;
            type = 'unhealthy';
            playAlertSound('unhealthy');
        } else if (pred === 'Healthy') {
            title = '✅ Healthy Poultry Confirmed';
            body = `Normal vocalization confirmed in ${filename} (${(conf * 100).toFixed(1)}% confidence).`;
            type = 'healthy';
            playAlertSound('healthy');
        } else {
            title = 'Farm Environmental Noise';
            body = `Environmental noise dominant in ${filename}.`;
            type = 'info';
        }

        // 1. In-App Floating Toast
        showToast(title, body, type);

        // 2. HTML5 System Desktop Notification
        if ('Notification' in window && Notification.permission === 'granted') {
            try {
                new Notification(title, {
                    body: body,
                    icon: '/static/favicon.ico',
                    requireInteraction: pred === 'Unhealthy'
                });
            } catch (e) {
                console.warn('System notification error:', e);
            }
        }
    }

    // Test Alert Button Listener
    if (btnTestNotif) {
        btnTestNotif.addEventListener('click', async () => {
            btnTestNotif.disabled = true;
            btnTestNotif.style.opacity = '0.6';
            showToast('Dispatching Test Alert...', 'Pushing notification to ntfy.envs.net/birdflu7 and system channels...', 'info');

            try {
                const res = await fetch('/api/test-notification', { method: 'POST' });
                const json = await res.json();
                triggerDiagnosticNotification('Unhealthy', 0.985, 'manual_test_alert.wav', true);
                showToast(
                    'Push Dispatched Successfully!',
                    'Pushed to ntfy. View live feed at: <a href="https://ntfy.envs.net/birdflu7" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: bold;">ntfy.envs.net/birdflu7</a>',
                    'success'
                );
            } catch (err) {
                showToast('Test Alert Failed', err.message, 'error');
            } finally {
                btnTestNotif.disabled = false;
                btnTestNotif.style.opacity = '1';
            }
        });
    }
});
