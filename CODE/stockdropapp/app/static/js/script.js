const runningDaemons = {};

let isDaemonRunning = false;

function startDaemon() {
    const currency = document.getElementById('currency').value;
    const time_period = document.getElementById('time_period').value;

    // Check if daemon is running for the current currency and time period
    if (runningDaemons[`${currency}_${time_period}`]) {
        // Daemon is running, so it can be disabled
        stopDaemon(`${currency}_${time_period}`);
    } else {
        // Daemon is not running, start it
        fetch(`/start_daemon/${currency}/${time_period}/Model1`)
            .then(response => response.text())
            .then(data => {
                alert(data);
                // Add the running daemon to the list
                runningDaemons[`${currency}_${time_period}`] = `Daemon for ${currency}, ${time_period} running...`;
                updateRunningDaemons();
            });
    }
}

function updateRunningDaemons() {
    const runningDaemonsList = document.getElementById('running_daemons');
    runningDaemonsList.innerHTML = '';  // Clear previous entries

    for (const key in runningDaemons) {
        const li = document.createElement('li');
        li.innerText = runningDaemons[key];
        // const stopButton = document.createElement('button');
        // stopButton.innerText = 'Stop';
        // stopButton.onclick = () => stopDaemon(key);
        // li.appendChild(stopButton);
        runningDaemonsList.appendChild(li);
    }

    // Check if a daemon is running for the current currency and time period
    const currency = document.getElementById('currency').value;
    const time_period = document.getElementById('time_period').value;
    isDaemonRunning = !!runningDaemons[`${currency}_${time_period}`];
    const button = document.getElementById('notification_toggle');
    button.innerText = isDaemonRunning ? 'Disable Notification' : 'Enable Notification';
}


function stopDaemon(key) {
    // Implement logic to stop the daemon based on the key
    runningDaemons[key] = `Daemon for ${key} stopped.`;
    updateRunningDaemons();
}