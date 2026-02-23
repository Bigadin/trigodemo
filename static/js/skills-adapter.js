/**
 * Skills Adapter - communique avec /api/skills pour recuperer skills et categories.
 * Fallback sur une config par defaut (detection presence + humain) si le back n'est pas dispo.
 * S'adapte a ce qui existe cote backend pour la demo UI.
 */
(function () {
    'use strict';

    // Voir docs/SPEC_MODAL_CREATION_BENEFICES.md pour la spec complete (4 skills, categories, sous-types)
    const DEFAULT_SKILLS_CONFIG = {
        skills: [
            { key: 'detection', label: 'Detection', icon: '/static/assets_youn/SvIcons/SVGnew/Yclassify.svg', items: [{ id: 'detection_presence', label: 'Presence / Absence', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }, { id: 'detection_linecross', label: 'Franchissement ligne', icon: '/static/assets_youn/SvIcons/SVGnew/Ylinecross.svg' }, { id: 'detection_zone', label: 'Detection zone', icon: '/static/assets_youn/SvIcons/SVGnew/Yzonedetect.svg' }] },
            { key: 'counting', label: 'Comptage', icon: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg', items: [{ id: 'counting_people', label: 'Comptage personnes', icon: '/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg' }, { id: 'counting_objects', label: 'Comptage objets', icon: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg' }, { id: 'counting_zone', label: 'Comptage zone', icon: '/static/assets_youn/SvIcons/SVGnew/square-area-svgrepo-com.svg' }] },
            { key: 'heatmap', label: 'Heatmap', icon: '/static/assets_youn/SvIcons/SVGnew/Yheatmap.svg', items: [{ id: 'heatmap_density', label: 'Densite de flux', icon: '/static/assets_youn/SvIcons/SVGnew/grid-svgrepo-com.svg' }, { id: 'heatmap_presence', label: 'Heatmap presence', icon: '/static/assets_youn/SvIcons/SVGnew/Yheatmapdense.svg' }, { id: 'heatmap_trajectory', label: 'Heatmap trajectoires', icon: '/static/assets_youn/SvIcons/SVGnew/Ytraj.svg' }] },
            { key: 'quality', label: 'Qualite', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg', items: [{ id: 'quality_fissure', label: 'Fissure', icon: '/static/assets_youn/SvIcons/SVGnew/Yfissure.svg' }, { id: 'quality_humidity', label: 'Humidite', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg' }, { id: 'quality_check', label: 'Qualite generale', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg' }] }
        ],
        categories_by_skill: {
            detection: [
                { key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg', items: [{ id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }, { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' }, { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' }] },
                { key: 'transport', label: 'Transport', icon: '/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg', items: [{ id: 'voiture', label: 'Voiture', icon: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg' }, { id: 'velo', label: 'Velo', icon: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg' }, { id: 'public_transport', label: 'Transport public', icon: '/static/assets_youn/SvIcons/SVGnew/Ypublic%20transport.svg' }, { id: 'avion', label: 'Avion', icon: '/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg' }, { id: 'moto', label: 'Moto', icon: '/static/assets_youn/SvIcons/SVGnew/motorcycle.svg' }] }
            ],
            counting: [
                { key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg', items: [{ id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }, { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' }, { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' }] },
                { key: 'transport', label: 'Transport', icon: '/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg', items: [{ id: 'voiture', label: 'Voiture', icon: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg' }, { id: 'velo', label: 'Velo', icon: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg' }, { id: 'public_transport', label: 'Transport public', icon: '/static/assets_youn/SvIcons/SVGnew/Ypublic%20transport.svg' }, { id: 'avion', label: 'Avion', icon: '/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg' }, { id: 'moto', label: 'Moto', icon: '/static/assets_youn/SvIcons/SVGnew/motorcycle.svg' }] }
            ],
            heatmap: [
                { key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg', items: [{ id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' }, { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' }, { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' }] },
                { key: 'object', label: 'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg', items: [{ id: 'encombrement', label: 'Encombrement', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' }, { id: 'zone_encombre', label: 'Zone encombre', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' }] }
            ],
            quality: [
                { key: 'object', label: 'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg', items: [{ id: 'fissure', label: 'Fissure', icon: '/static/assets_youn/SvIcons/SVGnew/Yfissure.svg' }, { id: 'humidity', label: 'Humidite', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg' }, { id: 'qualitycheck', label: 'Qualite generale', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg' }] }
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
                        // Toujours utiliser la config complète (4 skills + toutes catégories) pour la création de bénéfices
                        const hasFullConfig = Array.isArray(data.skills) && data.skills.length >= 4
                            && data.categories_by_skill && Object.keys(data.categories_by_skill).length >= 4;
                        // À la création on propose TOUT : utiliser DEFAULT (spec complète) pour garantir 4 skills + toutes catégories/sous-types
                        skillsConfigCache = DEFAULT_SKILLS_CONFIG;
                        return skillsConfigCache;
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
