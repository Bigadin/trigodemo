/**
 * Skills Adapter — communique avec /api/skills pour récupérer skills et catégories.
 * Fallback sur une config par défaut (détection présence + humain) si le back n'est pas dispo.
 * S'adapte à ce qui existe côté backend pour la démo UI.
 */
(function () {
    'use strict';

    const DEFAULT_SKILLS_CONFIG = {
        skills: [
            {
                key: 'detection',
                label: 'Détection',
                icon: '/static/assets_youn/SvIcons/SVGnew/Yclassify.svg',
                items: [
                    { id: 'detection_presence', label: 'Détection présence absence', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }
                ]
            }
        ],
        categories_by_skill: {
            detection: [
                { key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg', items: [{ id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }] }
            ]
        }
    };

    let skillsConfigCache = null;
    let skillsLoadPromise = null;

    async function loadSkillsConfig() {
        if (skillsLoadPromise) return skillsLoadPromise;
        skillsLoadPromise = (async () => {
            try {
                const res = await fetch('/api/skills');
                if (res.ok) {
                    const data = await res.json();
                    if (data && (data.skills?.length || data.categories_by_skill)) {
                        skillsConfigCache = data;
                        return data;
                    }
                }
            } catch (e) {
                console.warn('[skills-adapter] /api/skills non disponible, fallback config par défaut:', e.message);
            }
            skillsConfigCache = DEFAULT_SKILLS_CONFIG;
            return skillsConfigCache;
        })();
        return skillsLoadPromise;
    }

    function getSkillsConfig() {
        return skillsConfigCache || DEFAULT_SKILLS_CONFIG;
    }

    function getBenefitSkillGroups() {
        const cfg = getSkillsConfig();
        return cfg.skills || DEFAULT_SKILLS_CONFIG.skills;
    }

    function getBenefitCategoryGroupsBySkill(skill) {
        const cfg = getSkillsConfig();
        const bySkill = cfg.categories_by_skill || {};
        const cats = bySkill[skill];
        if (Array.isArray(cats) && cats.length > 0) return cats;
        return [];
    }

    window.skillsAdapter = {
        load: loadSkillsConfig,
        getConfig: getSkillsConfig,
        getSkillGroups: getBenefitSkillGroups,
        getCategoryGroupsBySkill: getBenefitCategoryGroupsBySkill
    };
})();
