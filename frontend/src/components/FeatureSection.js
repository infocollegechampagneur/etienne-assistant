import React from 'react';
import { Card, CardContent } from './ui/card';

export const FeatureSection = ({ messageTypes, setActiveTab, setShowCorrectionModal }) => {
  return (
    <>
      {/* Features Cards */}
      <div className="lg:col-span-4 mt-8">
        <h3 className="text-2xl font-bold text-center text-gray-900 mb-8">Fonctionnalités principales</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {Object.entries(messageTypes).map(([key, type]) => (
            <Card key={key} className="bg-white/80 backdrop-blur-sm border-orange-100 hover:shadow-lg transition-shadow cursor-pointer" 
                  onClick={() => setActiveTab(key)}>
              <CardContent className="p-6 text-center">
                <div className="text-3xl mb-3">{type.icon}</div>
                <h4 className="font-semibold text-gray-900 mb-2">{type.title}</h4>
                <p className="text-sm text-gray-600">{type.description}</p>
              </CardContent>
            </Card>
          ))}
          {/* Carte Corriger un texte */}
          <Card 
            className="bg-gradient-to-br from-red-50 to-orange-50 border-red-200 hover:shadow-lg transition-shadow cursor-pointer ring-2 ring-red-100" 
            onClick={() => setShowCorrectionModal(true)}
            data-testid="correction-feature-card"
          >
            <CardContent className="p-6 text-center">
              <div className="text-3xl mb-3">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto text-red-600"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Corriger un texte</h4>
              <p className="text-sm text-gray-600">Correction conforme aux grilles MEQ avec pondérations personnalisables</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Info Sections */}
      <div className="lg:col-span-4 mt-12 space-y-8">
        
        {/* Upload de fichiers */}
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 border border-green-200">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            Nouveau : Analysez vos documents
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Comment utiliser :</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>- Cliquez sur l'icone trombone à côté du champ de message</li>
                <li>- Sélectionnez votre document (PDF, Word, Excel, etc.)</li>
                <li>- Posez votre question sur le contenu</li>
                <li>- Étienne analyse et répond en se basant sur votre fichier</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Exemples de demandes :</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>- "Résume ce manuel scolaire pour créer un plan de cours"</li>
                <li>- "Extrais les concepts clés de ce chapitre"</li>
                <li>- "Crée des questions d'examen basées sur ce document"</li>
                <li>- "Analyse ce plan d'intervention pour un élève HDAA"</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 p-3 bg-white/60 rounded-lg">
            <p className="text-sm text-gray-700">
              <strong>Formats supportés :</strong> PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), 
              Texte (.txt), CSV &bull; <strong>Taille max :</strong> 10MB
            </p>
          </div>
        </div>

        {/* Téléchargement de documents */}
        <div className="bg-gradient-to-r from-blue-50 to-orange-50 rounded-xl p-6 border border-orange-200">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            Téléchargement de documents
          </h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Comment ça marche :</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>- Posez votre question à Étienne</li>
                <li>- Des boutons de téléchargement apparaîtront sous les réponses</li>
                <li>- Choisissez le format : PDF, Word, PowerPoint ou Excel</li>
                <li>- Le document se télécharge automatiquement</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Exemples par rôle :</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>- <strong>Enseignant:</strong> "Plan de cours sur la Révolution tranquille"</li>
                <li>- <strong>TES:</strong> "Crée un plan d'intervention pour élève TDAH"</li>
                <li>- <strong>Orthopédagogue:</strong> "Évaluation diagnostique en lecture Sec 1"</li>
                <li>- <strong>Direction:</strong> "Projet éducatif pour conseil d'établissement"</li>
                <li>- <strong>Secrétariat:</strong> "Lettre aux parents - activité parascolaire"</li>
                <li>- <strong>Travailleur social:</strong> "Modèle rapport d'évaluation psychosociale"</li>
              </ul>
            </div>
          </div>
        </div>
        
        {/* Ce qu'Étienne peut faire */}
        <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-6 border border-orange-200">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            Ce qu'Étienne peut faire pour vous
          </h3>
          
          <RoleSection title="Enseignants" color="orange" items={[
            'Plans de cours (PFEQ)', 'Examens avec corrigés', 'SAÉ et activités',
            "Grilles d'évaluation critériées", 'Exercices différenciés', 'Présentations PowerPoint'
          ]} />
          <RoleSection title="TES, Orthopédagogues, Intervenants psychosociaux" color="blue" items={[
            "Plans d'intervention (PI)", "Fiches d'observation", 'Stratégies TDAH, TSA, dyslexie',
            'Rapports de suivi', 'Évaluations diagnostiques', 'Outils de gestion de crise'
          ]} />
          <RoleSection title="Travailleurs sociaux" color="purple" items={[
            'Évaluations psychosociales', 'Rapports DPJ (LPJ)', 'Références ressources',
            'Documentation signalements', 'Suivis confidentiels', 'Interventions de groupe'
          ]} />
          <RoleSection title="Direction et Administration" color="green" items={[
            'Projets éducatifs', 'Plans stratégiques', 'Politiques institutionnelles',
            'Communications officielles', 'Présentations CA', 'Rapports annuels'
          ]} />
          <RoleSection title="Secrétariat et TOS" color="pink" items={[
            'Lettres officielles', 'Communications parents', 'Formulaires (absences, autorisations)',
            'Procès-verbaux', 'Grilles-horaires', 'Procédures administratives'
          ]} />
          <div>
            <h4 className="font-semibold text-teal-700 mb-2 flex items-center gap-2">Autres membres du personnel</h4>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
              <span>- <strong>Tech. loisir:</strong> Planification événements</span>
              <span>- <strong>Surveillants:</strong> Protocoles, rapports incidents</span>
              <span>- <strong>Tech. labo:</strong> Procédures sécurité (SIMDUT)</span>
              <span>- <strong>Coord. comm.:</strong> Infolettres, médias sociaux</span>
              <span>- <strong>Finance:</strong> Budgets, rapports financiers</span>
              <span>- <strong>Admissions:</strong> Formulaires, statistiques</span>
            </div>
          </div>
        </div>
      </div>
        
      {/* Image Section */}
      <div className="lg:col-span-4 mt-8 text-center">
        <img 
          src="https://images.unsplash.com/photo-1596574027151-2ce81d85af3e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NTY2NzB8MHwxfHNlYXJjaHw0fHxlZHVjYXRpb24lMjBsZWFybmluZ3xlbnwwfHx8fDE3NTk0MTA1OTh8MA&ixlib=rb-4.1.0&q=85" 
          alt="Environnement d'apprentissage" 
          className="w-full max-w-2xl mx-auto rounded-xl shadow-lg object-cover h-64"
        />
        <p className="text-gray-600 mt-4 italic">Étienne - Assistant IA pour tout le personnel scolaire québécois</p>
      </div>
    </>
  );
};

const RoleSection = ({ title, color, items }) => (
  <div className="mb-6">
    <h4 className={`font-semibold text-${color}-700 mb-2 flex items-center gap-2`}>{title}</h4>
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm text-gray-600">
      {items.map((item, i) => <span key={i}>- {item}</span>)}
    </div>
  </div>
);
