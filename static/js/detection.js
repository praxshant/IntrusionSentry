// Zone utility functions
class ZoneManager {
    constructor(points) {
        this.points = points;  // Array of [x, y] coordinates
    }

    // Point-in-polygon ray casting algorithm
    isPointInZone(x, y) {
        let inside = false;
        for (let i = 0, j = this.points.length - 1; i < this.points.length; j = i++) {
            const xi = this.points[i][0], yi = this.points[i][1];
            const xj = this.points[j][0], yj = this.points[j][1];
            
            const intersect = ((yi > y) !== (yj > y)) &&
                            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            
            if (intersect) {
                inside = !inside;
            }
        }
        return inside;
    }

    // Check if bounding box center is in zone
    isBoxInZone(box) {
        const [x1, y1, x2, y2] = box;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        return this.isPointInZone(cx, cy);
    }
}

// Person tracking for entry/exit detection
class PersonTracker {
    constructor(entryFrames = 8, exitFrames = 12) {
        this.slots = {};  // slot_id -> { present: bool, presentCount: int, absentCount: int }
        this.entryThreshold = entryFrames;
        this.exitThreshold = exitFrames;
    }

    updateSlot(slotId, isPresent) {
        if (!this.slots[slotId]) {
            this.slots[slotId] = {
                present: false,
                presentCount: 0,
                absentCount: 0
            };
        }
        
        const slot = this.slots[slotId];
        
        if (isPresent) {
            slot.presentCount = Math.min(this.entryThreshold, slot.presentCount + 1);
            slot.absentCount = 0;
        } else {
            slot.absentCount = Math.min(this.exitThreshold, slot.absentCount + 1);
            slot.presentCount = 0;
        }
        
        // Check for state change
        let eventType = null;
        
        if (!slot.present && slot.presentCount >= this.entryThreshold) {
            slot.present = true;
            eventType = 'entry';
        }
        
        if (slot.present && slot.absentCount >= this.exitThreshold) {
            slot.present = false;
            eventType = 'exit';
        }
        
        return eventType;
    }
}

// Canvas drawing utilities
class CanvasDrawer {
    constructor(ctx) {
        this.ctx = ctx;
    }

    drawBox(box, color = '#00ff00', lineWidth = 3) {
        const [x1, y1, x2, y2] = box;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    drawLabel(text, x, y, color = '#00ff00') {
        this.ctx.fillStyle = color;
        this.ctx.font = 'bold 16px Arial';
        this.ctx.fillText(text, x, y - 10);
    }

    drawZone(points, color = '#00ff00', lineWidth = 2, dashed = false) {
        if (!points || points.length === 0) return;
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        if (dashed) this.ctx.setLineDash([10, 5]);
        this.ctx.beginPath();
        this.ctx.moveTo(points[0][0], points[0][1]);
        
        for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i][0], points[i][1]);
        }
        
        this.ctx.closePath();
        this.ctx.stroke();
        if (dashed) this.ctx.setLineDash([]);
    }
}

// API communication helper
class APIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async matchFrame(imageBase64, box) {
        try {
            const response = await fetch(`${this.baseUrl}/match_frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageBase64, box: box })
            });
            
            if (!response.ok) {
                console.error('Frame matching failed:', response.status);
                return null;
            }
            
            return await response.json();
        } catch (error) {
            console.error('Frame matching error:', error);
            return null;
        }
    }

    async reportEvent(eventType, slot, label, screenshot) {
        try {
            const response = await fetch(`${this.baseUrl}/report_event`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    event_type: eventType,
                    person_slot: slot,
                    person_label: label,
                    screenshot: screenshot
                })
            });
            
            if (!response.ok) {
                console.error('Event reporting failed:', response.status);
                return null;
            }
            
            return await response.json();
        } catch (error) {
            console.error('Event reporting error:', error);
            return null;
        }
    }

    async getStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/status`);
            return await response.json();
        } catch (error) {
            console.error('Status fetch error:', error);
            return null;
        }
    }

    async getEvents() {
        try {
            const response = await fetch(`${this.baseUrl}/events`);
            return await response.json();
        } catch (error) {
            console.error('Events fetch error:', error);
            return null;
        }
    }
}

// Export for use in browser.html
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ZoneManager, PersonTracker, CanvasDrawer, APIClient };
}
