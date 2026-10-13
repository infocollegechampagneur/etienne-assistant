/**
 * PrivacyModal.js
 * Déclaration de confidentialité honnête et transparente d'Étienne.
 */

import React from 'react';

const Section = ({ icon, title, children }) => (
  <div className="mb-4">
    <h3 className="font-semibold text-gray-800 text-sm mb-1.5 flex items-center gap-2">
      <span>{icon}</span> {title}
    </h3>
    <div className="text-sm text-gray-600 leading-relaxed pl-7">{children}</div>
  </div>
);

const PrivacyModal = ({ onClose }) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        data-testid="privacy-modal"
        className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white border-b border-gray-100 px-6 py-4 flex items-center justify-between rounded-t-xl">
          <h2 className="text-lg font-bold text-gray-900">🔒 Confidentialité et protection des données</h2>
          <button
            data-testid="privacy-modal-close"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 text-xl leading-none transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="px-6 py-4">
          <p className="text-sm text-gray-700 mb-4 bg-blue-50 border border-blue-100 rounded-lg p-3">
            Cette page explique en toute transparence ce qu'Étienne fait — et ne fait pas — avec vos
            données et celles de vos élèves, dans l'esprit de la <strong>Loi 25 du Québec</strong>.
          </p>

          <Section icon="🛡️" title="Anonymisation automatique des élèves">
            Avant chaque envoi à l'intelligence artificielle, Étienne <strong>masque automatiquement
            le nom de l'élève</strong> (remplacé par « ÉLÈVE-ANONYME »), ainsi que
            <strong> tous les noms d'élèves de vos listes de classe</strong> mentionnés dans le texte
            (amis, camarades, etc.). Les vrais noms sont restaurés uniquement dans la correction
            affichée à l'enseignant. <strong>Les IA tierces ou le web ne voient jamais le nom de vos élèves.</strong>
          </Section>

          <Section icon="🔐" title="Informations sensibles masquées automatiquement">
            Sont aussi détectés et masqués avant tout envoi : les <strong>courriels</strong>, les
            <strong> numéros de téléphone</strong>, les <strong>adresses civiques</strong>, les
            <strong> codes postaux</strong>, les <strong>dates de naissance</strong>, les
            <strong> codes permanents du Québec</strong>, les <strong>numéros d'assurance maladie
            (NAM)</strong> et les <strong>numéros d'assurance sociale (NAS)</strong>.
          </Section>

          <Section icon="📁" title="Vos fichiers ne sont jamais conservés">
            Les fichiers téléversés (textes, photos, documents) sont traités <strong>en mémoire
            seulement</strong>, puis immédiatement détruits. Aucun fichier n'est sauvegardé sur nos
            serveurs ni dans la base de données.
          </Section>

          <Section icon="🚫" title="Aucun traqueur, aucune publicité">
            Étienne n'utilise aucun outil de suivi publicitaire (pas de Google Analytics, pas de
            cookies tiers, pas de revente de données). Vos conversations sont conservées dans
            <strong> notre propre base de données sécurisée</strong>, accessibles uniquement par votre compte.
          </Section>

          <Section icon="✅" title="Détection de texte IA : 100 % locale">
            L'analyse qui détecte si un texte a été écrit par une IA est effectuée
            <strong> entièrement sur nos serveurs</strong>. Aucun texte d'élève n'est envoyé à un
            service tiers pour cette fonction.
          </Section>

          <Section icon="☁️" title="Ce qui est envoyé au web (nécessaire au fonctionnement)">
            Pour corriger un texte ou lire une photo manuscrite, Étienne doit l'envoyer à l'API
            fourni par Étienne. Le texte est envoyé <strong>anonymisé</strong> (voir ci-dessus). Les photos
            manuscrites sont envoyées telles quelles pour la lecture : nous recommandons de
            <strong> plier ou masquer le nom de l'élève sur la feuille avant de la photographier</strong>.
          </Section>

          <Section icon="🔑" title="Transparence sur les clés API">
            Avec les <strong>clés API</strong>, les conditions des agents du WEB et IA tierces indiquent que les contenus
            soumis peuvent être utilisés pour améliorer leurs modèles d'IA. C'est précisément pourquoi
            Étienne anonymise les données avant l'envoi.
          </Section>

          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-800">
            <strong>En résumé :</strong> aucun nom d'élève, courriel ou téléphone ne quitte Étienne ;
            aucun fichier n'est conservé ; aucune donnée n'est vendue ni partagée à des fins
            publicitaires. Seul le contenu anonymisé transite par les agents du WEB et IA tierces pour permettre la correction.
          </div>

          <p className="text-xs text-gray-400 mt-4 text-center">
            Dernière mise à jour : juin 2026 — Étienne, Collège Champagneur
          </p>
        </div>
      </div>
    </div>
  );
};

export default PrivacyModal;
