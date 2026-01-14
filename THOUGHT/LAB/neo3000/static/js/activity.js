// ===== ACTIVITY FEED =====

export function addActivity(activity) {
    const feed = document.getElementById('activity-feed');

    // Remove placeholder
    if (feed.children.length === 1 && feed.children[0].style.color) {
        feed.innerHTML = '';
    }

    const item = document.createElement('div');
    item.className = 'activity-item';

    const time = new Date(activity.timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });

    item.innerHTML = `
        <span class="activity-time">${time}</span>
        <span class="activity-badge ${activity.action}">${activity.action}</span>
        <span class="activity-text">${activity.summary}</span>
    `;

    feed.insertBefore(item, feed.firstChild);

    // Limit items
    while (feed.children.length > 20) {
        feed.removeChild(feed.lastChild);
    }
}
