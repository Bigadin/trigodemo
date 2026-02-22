/**
 * Utilitaires purs — formatage, échappement, etc.
 * Aucune dépendance. Exposé sur window.appUtils.
 */
(function () {
    'use strict';

    function pad2(n) {
        return String(n).padStart(2, '0');
    }

    function formatHMS(totalSeconds) {
        const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const ss = s % 60;
        return `${pad2(h)}:${pad2(m)}:${pad2(ss)}`;
    }

    function escapeHtml(text) {
        return String(text ?? '')
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#39;');
    }

    function truncateFilename(name, maxLen) {
        maxLen = maxLen ?? 20;
        if (!name || name.length <= maxLen) return name;
        const ext = name.lastIndexOf('.') > 0 ? name.slice(name.lastIndexOf('.')) : '';
        const base = name.slice(0, name.length - ext.length);
        const availableLen = maxLen - ext.length - 3;
        if (availableLen <= 0) return name.slice(0, maxLen - 3) + '...';
        return base.slice(0, availableLen) + '...' + ext;
    }

    function toLocalInputDateTime(date) {
        const d = new Date(date ?? new Date());
        const y = d.getFullYear();
        const m = pad2(d.getMonth() + 1);
        const day = pad2(d.getDate());
        const h = pad2(d.getHours());
        const min = pad2(d.getMinutes());
        return `${y}-${m}-${day}T${h}:${min}`;
    }

    function formatLogTimestamp(isoStr) {
        try {
            const d = new Date(isoStr);
            return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
        } catch {
            return isoStr;
        }
    }

    window.appUtils = {
        pad2,
        formatHMS,
        escapeHtml,
        truncateFilename,
        toLocalInputDateTime,
        formatLogTimestamp
    };
})();
