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

document.addEventListener('DOMContentLoaded', function () {
    // Function to update the plot
    function updatePlot() {
        // Get the selected currency, time period, and data type
        const selectedCurrency = document.getElementById('currency').value;
        const selectedTimePeriod = document.getElementById('time_period').value;
        const selectedData = document.getElementById('data_type').value;

        // Make an AJAX request to the server to fetch updated plot data
        fetch('/update_plot', {
            method: 'POST',
            body: JSON.stringify({
                currency: selectedCurrency,
                time_period: selectedTimePeriod,
                data_type: selectedData,
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then(response => response.json())
            .then(data => {
                // Update the Plotly plot with the new data
                const data_json = JSON.parse(data);
                const plotDiv = document.getElementById('plot-div');
                console.log(typeof data_json);
                var trace2 = {
                    x: data_json['x'].map(dateString => new Date(dateString)),
                    y: data_json['y'],
                    type: 'scatter'
                  };
                console.log(trace2);
                Plotly.newPlot(plotDiv, [trace2]);
            });
    }

    // Get the data type select box element
    const dataTypeSelect = document.getElementById('data_type');
    const currencySelect = document.getElementById('currency');
    const time_periodSelect = document.getElementById('time_period');

    // Add an event listener to capture the change event
    dataTypeSelect.addEventListener('change', updatePlot);
    currencySelect.addEventListener('change', updatePlot);
    time_periodSelect.addEventListener('change', updatePlot);

    // Initialize the plot when the page loads
    updatePlot();
});
