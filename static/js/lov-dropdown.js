/**
 * LovDropdown — Custom glassmorphism dropdown replacement for <select>
 *
 * Wraps a native <select> element: hides it and renders an animated
 * custom dropdown that stays in sync (bidirectionally).
 *
 * Usage:
 *   LovDropdown.wrap(selectElement)          — wrap one select
 *   LovDropdown.wrapAll(selector?)           — wrap all selects matching selector (default: 'select')
 *   LovDropdown.refresh(selectElement)       — rebuild options after innerHTML change
 *
 * The native <select> keeps working: .value, .innerHTML, dispatchEvent('change') etc.
 * are all synced through a MutationObserver + manual refresh fallback.
 */
(function () {
    'use strict';

    const CHEVRON_SVG = `<svg class="lov-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;

    const instances = new WeakMap();

    class LovDropdown {

        /** @param {HTMLSelectElement} select */
        constructor(select) {
            if (instances.has(select)) return instances.get(select);
            this.select = select;
            this._build();
            this._observe();
            instances.set(select, this);
        }

        /* ---- DOM construction ---- */

        _build() {
            const sel = this.select;

            // Container
            this.root = document.createElement('div');
            this.root.className = 'lov-dropdown';
            if (sel.style.width === '100%' || sel.classList.contains('lov-select'))
                this.root.classList.add('lov-dropdown--full');

            // Trigger
            this.trigger = document.createElement('button');
            this.trigger.type = 'button';
            this.trigger.className = 'lov-trigger';
            this.trigger.innerHTML = `<span class="lov-trigger-text"></span>${CHEVRON_SVG}`;
            this.triggerText = this.trigger.querySelector('.lov-trigger-text');

            // Panel
            this.panel = document.createElement('div');
            this.panel.className = 'lov-panel';

            this.root.appendChild(this.trigger);
            this.root.appendChild(this.panel);

            // Insert & hide native
            sel.parentNode.insertBefore(this.root, sel);
            sel.style.position = 'absolute';
            sel.style.opacity = '0';
            sel.style.pointerEvents = 'none';
            sel.style.width = '0';
            sel.style.height = '0';
            sel.style.overflow = 'hidden';
            sel.tabIndex = -1;
            this.root.appendChild(sel); // keep inside container for form compat

            this._renderOptions();
            this._updateLabel();

            // Events
            this.trigger.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggle();
            });

            // Close on outside click
            this._onDocClick = (e) => {
                if (!this.root.contains(e.target)) this.close();
            };
            document.addEventListener('click', this._onDocClick, true);

            // Close on Escape
            this._onKeyDown = (e) => {
                if (e.key === 'Escape') this.close();
            };
            document.addEventListener('keydown', this._onKeyDown);

            // Sync when native select changes programmatically
            sel.addEventListener('change', () => this._updateLabel());
        }

        _renderOptions() {
            const opts = Array.from(this.select.options);
            this.panel.innerHTML = '';

            if (opts.length === 0) {
                this.panel.innerHTML = '<div class="lov-option-empty">Aucune option</div>';
                return;
            }

            opts.forEach((opt, i) => {
                // Skip the placeholder-style options that are empty value
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'lov-option';
                btn.style.setProperty('--i', i);
                btn.dataset.value = opt.value;
                btn.textContent = opt.textContent;

                if (opt.value === this.select.value) {
                    btn.classList.add('is-selected');
                }

                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this._selectValue(opt.value);
                });

                this.panel.appendChild(btn);
            });
        }

        _updateLabel() {
            const sel = this.select;
            const opt = sel.options[sel.selectedIndex];
            if (opt && opt.value) {
                this.triggerText.textContent = opt.textContent;
                this.triggerText.classList.remove('placeholder');
            } else {
                // show placeholder from first empty-value option, or default
                const placeholder = sel.querySelector('option[value=""]');
                this.triggerText.textContent = placeholder ? placeholder.textContent : 'Sélectionner…';
                this.triggerText.classList.add('placeholder');
            }

            // Update selected highlight
            this.panel.querySelectorAll('.lov-option').forEach(btn => {
                btn.classList.toggle('is-selected', btn.dataset.value === sel.value);
            });
        }

        _selectValue(value) {
            this.select.value = value;
            this.select.dispatchEvent(new Event('change', { bubbles: true }));
            this._updateLabel();
            this.close();
        }

        /* ---- MutationObserver (catches innerHTML rebuilds) ---- */

        _observe() {
            this._observer = new MutationObserver(() => {
                this._renderOptions();
                this._updateLabel();
            });
            this._observer.observe(this.select, { childList: true, subtree: true, characterData: true });
        }

        /* ---- Public API ---- */

        toggle() {
            this.root.classList.contains('is-open') ? this.close() : this.open();
        }

        open() {
            this.root.classList.add('is-open');
            // Reset stagger animations by forcing re-render
            const options = this.panel.querySelectorAll('.lov-option');
            options.forEach(o => {
                o.style.opacity = '0';
                o.style.transform = 'translateX(8px) scale(0.96)';
                o.style.filter = 'blur(6px)';
            });
            // Trigger animation on next frame
            requestAnimationFrame(() => {
                options.forEach(o => {
                    o.style.opacity = '';
                    o.style.transform = '';
                    o.style.filter = '';
                });
            });
        }

        close() {
            this.root.classList.remove('is-open');
        }

        refresh() {
            this._renderOptions();
            this._updateLabel();
        }

        destroy() {
            document.removeEventListener('click', this._onDocClick, true);
            document.removeEventListener('keydown', this._onKeyDown);
            if (this._observer) this._observer.disconnect();
            // Restore native select
            const sel = this.select;
            this.root.parentNode.insertBefore(sel, this.root);
            sel.style.cssText = '';
            sel.tabIndex = 0;
            this.root.remove();
            instances.delete(sel);
        }

        /* ---- Static helpers ---- */

        /** Wrap a single <select> */
        static wrap(select) {
            if (!(select instanceof HTMLSelectElement)) return null;
            return new LovDropdown(select);
        }

        /** Wrap all <select> matching selector */
        static wrapAll(selector = 'select') {
            document.querySelectorAll(selector).forEach(sel => LovDropdown.wrap(sel));
        }

        /** Force refresh (useful after programmatic innerHTML changes) */
        static refresh(select) {
            const inst = instances.get(select);
            if (inst) inst.refresh();
        }

        /** Get instance for a select */
        static getInstance(select) {
            return instances.get(select) || null;
        }
    }

    // Expose globally
    window.LovDropdown = LovDropdown;

})();
