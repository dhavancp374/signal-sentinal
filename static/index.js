let map, marker;

// Initialize the Leaflet map
function initializeMap() {
    map = L.map('map').setView([13.7259, 80.2266], 64030800); // Initial position
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19
    }).addTo(map);
    marker = L.marker([13.7259, 80.2266]).addTo(map).bindPopup('Current Location').openPopup();
}

// Update marker position on the map
function updateMap(lat, lon) {
    marker.setLatLng([lat, lon]);
    map.setView([lat, lon]); // Center map to the new position
    marker.getPopup().setContent(`Current Location: [${lat.toFixed(6)}, ${lon.toFixed(6)}]`);
    marker.openPopup();
}

initializeMap(); // Initialize the map on page load

// Update GPS data periodically

const speedCtx = document.getElementById('speedChart').getContext('2d');
const speedData = {
    labels: [], // Dynamic time labels
    datasets: [{
        label: 'Speed (km/h)',
        data: [],
        borderColor: 'blue',
        tension: 0.1,
        fill: false
    }]
};
const speedChart = new Chart(speedCtx, {
    type: 'line',
    data: speedData,
    options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
            x: { title: { display: true, text: 'Time' } },
            y: { title: { display: true, text: 'Speed (km/h)' } }
        }
    }
});

// Pie chart for event types (Spoofing, Jamming, Normal)
const pieCtx = document.getElementById('eventChart').getContext('2d');

// Initial data for the chart (all values set to 0 initially)
const eventData = {
    labels: ['Spoofing', 'Normal'],
    datasets: [{
        data: [0, 0, 0], // Initial data values for Spoofing, Jamming, Normal
        backgroundColor: ['red','orange', 'green'], // Red = Spoofing, Orange = Jamming, Green = Normal
        borderColor: ['white'],
        borderWidth: 2
    }]
};

// Create the pie chart
const eventChart = new Chart(pieCtx, {
    type: 'pie',
    data: eventData,
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            tooltip: {
                callbacks: {
                    label: tooltipItem => `${tooltipItem.label}: ${tooltipItem.raw}%`
                }
            },
            datalabels: {
                formatter: (value, ctx) => `${value}%`
            }
        }
    }
});

// Navigation (No changes)
document.getElementById('dashboard-link').onclick = () => showSection('dashboard');
document.getElementById('map-link').onclick = () => showSection('map');
document.getElementById('alerts-link').onclick = () => showSection('alerts');
function showSection(section) {
    document.querySelectorAll('.content-section').forEach(el => el.style.display = 'none');
    document.getElementById(`${section}-content`).style.display = 'block';
}
showSection('dashboard'); // Default view

// Initialize counters for anomalies
let spoofingCount = 0, jammingCount = 0, normalCount = 0;

// Function to update the GPS and anomaly data
async function updateGPSData() {
    const gpsResponse = await fetch('/gps-data');
    const gpsData = await gpsResponse.json();

    const anomalyResponse = await fetch('/detect-anomaly', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gpsData)
    });
    const anomalyData = await anomalyResponse.json();
    // Update alerts if anomaly is detected
    if (anomalyData.is_anomaly || anomalyData.is_jamming) {
        const alertContainer = document.getElementById('alerts');
        const alert = document.createElement('p');
        alert.classList.add(anomalyData.is_anomaly ? 'alert' : 'jamming-alert');
        alert.innerText = anomalyData.alert_message;
        alertContainer.appendChild(alert);
    }
    document.getElementById('latitude').innerText = gpsData.lat.toFixed(6);
    document.getElementById('longitude').innerText = gpsData.lon.toFixed(6);
    document.getElementById('speed').innerText = gpsData.speed.toFixed(2);
    updateMap(gpsData.lat, gpsData.lon)
    const now = new Date().toLocaleTimeString();
    speedData.labels.push(now);
    speedData.datasets[0].data.push(gpsData.speed);
    if (speedData.labels.length > 10) { // Limit to 10 data points
        speedData.labels.shift();
        speedData.datasets[0].data.shift();
    }
    speedChart.update();

    if (anomalyData.is_anomaly) {
        spoofingCount++;
    } else if (anomalyData.is_jamming) {
        jammingCount++;
    } else {
        normalCount++;
    }


    const total = spoofingCount + jammingCount + normalCount;

    if (total === 0) return;

    const spoofingPercentage = ((spoofingCount / total) * 100).toFixed(1);
    const jammingPercentage = ((jammingCount / total) * 100).toFixed(1);
    const normalPercentage = (100 - parseFloat(spoofingPercentage) - parseFloat(jammingPercentage)).toFixed(1);

    eventData.datasets[0].data = [spoofingPercentage, jammingPercentage, normalPercentage];
    eventChart.update();
}

setInterval(updateGPSData, 2000);
