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

// PLOTS SECTION

document.addEventListener('DOMContentLoaded', function () {
    // Function to update the plot
    function updatePlots() {
        // Get the selected currency, time period, and data type
        const selectedCurrency = document.getElementById('currency').value;
        const selectedTimePeriod = document.getElementById('time_period').value;
        const selectedData = document.getElementById('data_type').value;

        // Make an AJAX request to the server to fetch updated plot data
        fetch('/update_plots', {
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
                // Update the Plotly plots with the new data
                const data_json = JSON.parse(data);
                var plotDiv = document.getElementById('plot1');
                // First plot
                var trace = {
                    x: data_json['x'].map(dateString => new Date(dateString)),
                    y: data_json['y'],
                    type: 'scatter'
                  };
                var data_plot = [trace];
                var layout = {
                    title: 'Stock Price - Basic Plot'
                  };
                Plotly.newPlot(plotDiv, data_plot, layout);
                // Second plot
                var plotDiv = document.getElementById('plot2');
                var trace = {
                    x: data_json['x'].map(dateString => new Date(dateString)),
                    y: data_json['y'],
                    type: 'scatter'
                  };
                var data_plot = [trace];
                var layout = {
                    title: 'Stock Price - Basic Plot 2'
                  };
                Plotly.newPlot(plotDiv, data_plot, layout);
                // Third plot
                var plotDiv = document.getElementById('plot3');
                var trace = {
                    x: data_json['x'].map(dateString => new Date(dateString)),
                    y: data_json['y'],
                    type: 'scatter'
                  };
                var data_plot = [trace];
                var layout = {
                    title: 'Stock Price - Basic Plot 3'
                  };
                Plotly.newPlot(plotDiv, data_plot, layout);
                // Fourth plot
                var plotDiv = document.getElementById('plot4');
                var trace = {
                    x: data_json['x'].map(dateString => new Date(dateString)),
                    y: data_json['y'],
                    type: 'scatter'
                  };
                var data_plot = [trace];
                var layout = {
                    title: 'Stock Price - Basic Plot 4'
                  };
                Plotly.newPlot(plotDiv, data_plot, layout);
            });
    }

    // Get the data type select box element
    const dataTypeSelect = document.getElementById('data_type');
    const currencySelect = document.getElementById('currency');
    const time_periodSelect = document.getElementById('time_period');

    // Add an event listener to capture the change event
    dataTypeSelect.addEventListener('change', updatePlots);
    currencySelect.addEventListener('change', updatePlots);
    time_periodSelect.addEventListener('change', updatePlots);

    // Initialize the plot when the page loads
    updatePlots();
});

//NEWS SECTION

document.addEventListener('DOMContentLoaded', function () {
    // Function to update the plot
    function updateNews() {
        // Get the selected currency, time period, and data type
        const selectedCurrency = document.getElementById('currency').value;

        // Make an AJAX request to the server to fetch updated plot data
        fetch('/update_news', {
            method: 'POST',
            body: JSON.stringify({
                currency: selectedCurrency,
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then(response => response.json())
            .then(data => {
                // Update the Plotly plots with the new data
                const data_json = JSON.parse(data);
                console.log(data_json);
                // for key in data_json write news info in table format in index.html id news_area
                var table = document.getElementById("news_table_body");
                table.innerHTML = '';
                for (var i = 0; i < 5; i++) {
                    var row = table.insertRow(i);
                    var cell1 = row.insertCell(0);
                    var cell2 = row.insertCell(1);
                    var cell3 = row.insertCell(2);
                    var cell4 = row.insertCell(3);
                    var cell5 = row.insertCell(4);
                    var cell6 = row.insertCell(5);
                    var cell7 = row.insertCell(6);
                    var cell8 = row.insertCell(7);
                    var cell9 = row.insertCell(8);
                    var cell10 = row.insertCell(9);
                    var cell11 = row.insertCell(10);
                    var cell12 = row.insertCell(11);
                    // uuid,title,description,keywords,snippet,url,image_url,language,published_at,source,relevance_score,type,industry,match_score,sentiment_score
                    cell1.innerHTML = '<a href="' + data_json['url'][i] + '" target="_blank">' + data_json['title'][i] + '</a>'
                    cell2.innerHTML = data_json['description'][i];
                    cell3.innerHTML = data_json['keywords'][i];
                    cell4.innerHTML = data_json['snippet'][i];
                    cell5.innerHTML = data_json['language'][i];
                    cell6.innerHTML = data_json['published_at'][i];
                    cell7.innerHTML = data_json['source'][i];
                    cell8.innerHTML = data_json['relevance_score'][i];
                    cell9.innerHTML = data_json['type'][i];
                    cell10.innerHTML = data_json['industry'][i];
                    cell11.innerHTML = data_json['match_score'][i];
                    cell12.innerHTML = data_json['sentiment_score'][i];
                }
            });
    }

    // Get the data type select box element
    const currencySelect = document.getElementById('currency');

    // Add an event listener to capture the change event
    currencySelect.addEventListener('change', updateNews);

    // Initialize the plot when the page loads
    updateNews();
});