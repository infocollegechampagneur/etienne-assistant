import React, { useState, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Textarea } from './components/ui/textarea';
import { Checkbox } from './components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Separator } from './components/ui/separator';
import StudentProgressModal from './StudentProgressModal';

const PONDERATIONS = {
  'sec1-4': { C1: 25, C2: 20, C3: 10, C4: 25, C5: 20 },
  'sec5':   { C1: 30, C2: 20, C3: 5,  C4: 25, C5: 20 }
};

const CRITERES_LABELS = {
  C1: 'Adaptation à la situation de communication',
  C2: 'Cohérence du texte',
  C3: 'Vocabulaire approprié',
  C4: 'Syntaxe et ponctuation',
  C5: 'Orthographe (usage et grammaticale)'
};

// ─── GENRES DE TEXTES + GRILLES SPÉCIFIQUES (inspirées des grilles Tshakapesh) ───
const GENRES_TEXTE = [
  { value: 'descriptif', label: 'Texte descriptif' },
  { value: 'narratif', label: 'Texte narratif (récit)' },
  { value: 'explicatif', label: 'Texte explicatif' },
  { value: 'argumentatif', label: 'Texte argumentatif' },
  { value: 'justificatif', label: 'Texte justificatif' },
  { value: 'poetique', label: 'Texte poétique / poème' },
  { value: 'slam', label: 'Slam' },
  { value: 'fantastique', label: 'Récit fantastique' },
  { value: 'conte', label: 'Conte / légende' },
  { value: 'nouvelle', label: 'Nouvelle littéraire' },
  { value: 'lettre_ouverte', label: 'Lettre ouverte' },
  { value: 'article', label: 'Article de journal' },
  { value: 'autre', label: 'Autre (préciser)' }
];

const GRILLES_GENRE = {
  descriptif: `- Sujet clairement présenté dès l'introduction
- Aspects et sous-aspects du sujet développés et bien choisis
- Progression logique (ordre des aspects, organisateurs textuels)
- Vocabulaire précis et varié lié au sujet décrit
- Structure : introduction, développement (un aspect par paragraphe), conclusion`,
  narratif: `- Schéma narratif complet : situation initiale, élément déclencheur, péripéties, dénouement, situation finale
- Univers narratif cohérent (personnages, temps, lieux crédibles et maintenus)
- Narrateur choisi (interne/externe) et maintenu tout au long du récit
- Harmonisation des temps du récit (passé simple/imparfait OU présent)
- Personnages caractérisés (physique, psychologie, actions)`,
  explicatif: `- Phénomène clairement posé (question en POURQUOI ou COMMENT)
- Procédés explicatifs variés : définition, exemple, comparaison, reformulation, cause-conséquence
- Progression logique des explications (organisateurs textuels)
- Objectivité et neutralité (pas d'opinion personnelle)
- Vocabulaire précis, termes techniques expliqués`,
  argumentatif: `- Thèse clairement énoncée
- Arguments développés et appuyés (faits, exemples, statistiques, autorités)
- Procédés argumentatifs : réfutation, concession, question rhétorique
- Stratégie argumentative cohérente et maintenue
- Conclusion qui réaffirme la thèse`,
  justificatif: `- Affirmation/appréciation clairement énoncée
- Raisons pertinentes et variées appuyées d'exemples ou d'extraits
- Liens explicites entre les raisons et l'affirmation
- Vocabulaire appréciatif et justificatif approprié
- Organisation claire (une raison par paragraphe)`,
  poetique: `- Procédés stylistiques : images, comparaisons, métaphores, personnifications
- Musicalité : rythme, sonorités, rimes ou vers libres assumés
- Thème exploité avec originalité et sensibilité
- Mise en page volontaire (strophes, vers, disposition)
⚠️ ADAPTATION C4 : la syntaxe poétique permet des licences (phrases non verbales, inversions, absence de ponctuation ASSUMÉE) — ne pas pénaliser les choix stylistiques volontaires, seulement les erreurs involontaires.`,
  slam: `- Rythme et oralité perceptibles à la lecture
- Procédés sonores : allitérations, assonances, répétitions, anaphores
- Force du message, engagement, authenticité de la voix
- Registre de langue assumé et cohérent avec le propos
⚠️ ADAPTATION C4 : le slam permet des licences poétiques (syntaxe éclatée, phrases non verbales, registre familier VOLONTAIRE) — ne pénaliser que les erreurs involontaires (orthographe, accords).`,
  fantastique: `- Récit ancré dans le RÉEL où surgit un élément surnaturel inexpliqué
- Hésitation/doute maintenu (le lecteur ne sait pas si c'est réel ou surnaturel)
- Atmosphère inquiétante (champ lexical de la peur, du doute, de l'étrange)
- Narrateur souvent interne (je) pour renforcer le doute
- Schéma narratif complet et gradation de la tension`,
  conte: `- Formules d'ouverture et de fermeture (Il était une fois...)
- Merveilleux assumé (magie, créatures, objets enchantés acceptés d'emblée)
- Personnages types (héros, opposant, adjuvant)
- Schéma narratif complet, quête ou mission
- Morale ou leçon (explicite ou implicite)`,
  nouvelle: `- Récit bref et efficace (économie de moyens)
- Chute surprenante ou révélatrice à la fin
- Personnages peu nombreux, campés rapidement
- Intrigue resserrée (un seul fil narratif)
- Atmosphère et point de vue narratif maintenus`,
  lettre_ouverte: `- Destinataire explicite et interpellé
- Prise de position claire dès le début
- Arguments développés et appuyés
- Ton adapté (engagé mais respectueux), marques d'énonciation (je/nous/vous)
- Formules d'usage (appel, salutation, signature)`,
  article: `- Titre accrocheur et chapeau résumant l'essentiel
- Réponses aux questions : qui, quoi, où, quand, comment, pourquoi
- Objectivité (faits vérifiables, pas d'opinion sauf éditorial)
- Citations et sources rapportées
- Structure en pyramide inversée (essentiel d'abord)`
};

// ─── BARÈMES OFFICIELS CS LAVAL / MEQ (pages 14-22) ───
const WORD_RANGES = [
  '101-125','126-150','151-175','176-200','201-225','226-250',
  '251-275','276-300','301-325','326-350','351-375','376-400',
  '401-425','426-450','451-475','476-500','501+'
];

const BAREMES_COMPLETS = {
  sec1: {
    C4: {
      '101-125':['0-2','3-4','5-6','7-9','10+'],'126-150':['0-3','4-5','6-8','9-10','11+'],
      '151-175':['0-3','4-6','7-9','10-12','13+'],'176-200':['0-3','4-7','8-10','11-14','15+'],
      '201-225':['0-4','5-8','9-11','12-15','16+'],'226-250':['0-4','5-8','9-13','14-17','18+'],
      '251-275':['0-5','6-9','10-14','15-19','20+'],'276-300':['0-5','6-10','11-15','16-21','22+'],
      '301-325':['0-6','7-11','12-16','17-22','23+'],'326-350':['0-6','7-12','13-18','19-24','25+'],
      '351-375':['0-6','7-13','14-19','20-26','27+'],'376-400':['0-7','8-14','15-20','21-27','28+'],
      '401-425':['0-7','8-14','15-22','23-29','30+'],'426-450':['0-8','9-15','16-23','24-31','32+'],
      '451-475':['0-8','9-16','17-24','25-32','33+'],'476-500':['0-9','10-17','18-25','26-33','34+'],
      '501+':['0-9','10-17','18-25','26-34','35+']
    },
    C5: {
      '101-125':['0-4','5-7','8-10','11-14','15+'],'126-150':['0-5','6-8','9-12','13-17','18+'],
      '151-175':['0-5','6-9','10-14','15-20','21+'],'176-200':['0-6','7-11','12-15','16-23','24+'],
      '201-225':['0-7','8-12','13-17','18-26','27+'],'226-250':['0-8','9-13','14-19','20-29','30+'],
      '251-275':['0-8','9-15','16-21','22-32','33+'],'276-300':['0-9','10-16','17-23','24-34','35+'],
      '301-325':['0-10','11-18','19-25','26-37','38+'],'326-350':['0-11','12-19','20-27','28-40','41+'],
      '351-375':['0-11','12-20','21-29','30-43','44+'],'376-400':['0-12','13-22','23-31','32-46','47+'],
      '401-425':['0-13','14-23','24-33','34-49','50+'],'426-450':['0-14','15-24','25-35','36-52','53+'],
      '451-475':['0-14','15-26','27-37','38-54','55+'],'476-500':['0-15','16-27','28-39','40-57','58+'],
      '501+':['0-15','16-27','28-39','40-57','58+']
    }
  },
  sec2: {
    C4: {
      '101-125':['0-2','3-4','5-6','7','8+'],'126-150':['0-2','3-4','5-7','8-9','10+'],
      '151-175':['0-3','4-5','6-8','9-10','11+'],'176-200':['0-3','4-6','7-9','10-12','13+'],
      '201-225':['0-3','4-7','8-10','11-13','14+'],'226-250':['0-4','5-7','8-11','12-15','16+'],
      '251-275':['0-4','5-8','9-12','13-16','17+'],'276-300':['0-4','5-9','10-13','14-18','19+'],
      '301-325':['0-5','6-10','11-15','16-19','20+'],'326-350':['0-5','6-10','11-16','17-21','22+'],
      '351-375':['0-6','7-11','12-17','18-22','23+'],'376-400':['0-6','7-12','13-18','19-24','25+'],
      '401-425':['0-6','7-13','14-19','20-25','26+'],'426-450':['0-7','8-13','14-20','21-27','28+'],
      '451-475':['0-7','8-14','15-21','22-28','29+'],'476-500':['0-7','8-15','16-22','23-30','31+'],
      '501+':['0-7','8-15','16-22','23-30','31+']
    },
    C5: {
      '101-125':['0-3','4-6','7-8','9-12','13+'],'126-150':['0-4','5-7','8-10','11-14','15+'],
      '151-175':['0-4','5-8','9-11','12-17','18+'],'176-200':['0-5','6-9','10-13','14-19','20+'],
      '201-225':['0-6','7-10','11-15','16-21','22+'],'226-250':['0-6','7-11','12-16','17-24','25+'],
      '251-275':['0-7','8-12','13-18','19-26','27+'],'276-300':['0-7','8-13','14-19','20-28','29+'],
      '301-325':['0-8','9-15','16-21','22-31','32+'],'326-350':['0-9','10-16','17-23','24-33','34+'],
      '351-375':['0-9','10-17','18-24','25-36','37+'],'376-400':['0-10','11-18','19-26','27-38','39+'],
      '401-425':['0-11','12-19','20-28','29-40','41+'],'426-450':['0-11','12-20','21-29','30-43','44+'],
      '451-475':['0-12','13-21','22-31','32-45','46+'],'476-500':['0-12','13-22','23-32','33-47','48+'],
      '501+':['0-12','13-22','23-32','33-47','48+']
    }
  },
  sec3: {
    C4: {
      '101-125':['0-1','2-3','4-5','6','7+'],'126-150':['0-1','2-3','4-5','6-7','8+'],
      '151-175':['0-2','3-5','6-7','8','9+'],'176-200':['0-2','3-5','6-7','8-9','10+'],
      '201-225':['0-3','4-6','7-9','10-11','12+'],'226-250':['0-3','4-6','7-10','11-12','13+'],
      '251-275':['0-3','4-7','8-11','12-13','14+'],'276-300':['0-3','4-8','9-12','13-15','16+'],
      '301-325':['0-4','5-8','9-14','15-16','17+'],'326-350':['0-4','5-8','9-14','15-17','18+'],
      '351-375':['0-4','5-10','11-15','16-19','20+'],'376-400':['0-5','6-10','11-16','17-20','21+'],
      '401-425':['0-5','6-10','11-17','18-21','22+'],'426-450':['0-5','6-12','13-18','19-22','23+'],
      '451-475':['0-6','7-12','13-19','20-24','25+'],'476-500':['0-6','7-12','13-20','21-25','26+'],
      '501+':['0-6','7-12','13-20','21-25','26+']
    },
    C5: {
      '101-125':['0-2','3-4','5-7','8-9','10+'],'126-150':['0-2','3-5','6-8','9-10','11+'],
      '151-175':['0-3','4-5','6-9','10-12','13+'],'176-200':['0-3','4-7','8-10','11-14','15+'],
      '201-225':['0-4','5-8','9-12','13-16','17+'],'226-250':['0-4','5-9','10-13','14-18','19+'],
      '251-275':['0-5','6-10','11-14','15-20','21+'],'276-300':['0-5','6-11','12-16','17-22','23+'],
      '301-325':['0-6','7-12','13-17','18-24','25+'],'326-350':['0-6','7-13','14-18','19-26','27+'],
      '351-375':['0-7','8-13','14-20','21-28','29+'],'376-400':['0-7','8-14','15-21','22-29','30+'],
      '401-425':['0-8','9-15','16-21','22-31','32+'],'426-450':['0-8','9-15','16-24','25-33','34+'],
      '451-475':['0-9','10-16','17-24','25-35','36+'],'476-500':['0-9','10-17','18-25','26-37','38+'],
      '501+':['0-10','11-18','19-26','27-37','38+']
    }
  },
  sec4: {
    C4: {
      '101-125':['0-1','2','3','4','5+'],'126-150':['0-1','2-3','4-5','6','7+'],
      '151-175':['0-2','3-4','5-6','7','8+'],'176-200':['0-2','3-4','5-6','7-8','9+'],
      '201-225':['0-2','3-5','6-7','8-9','10+'],'226-250':['0-2','3-5','6-8','9-10','11+'],
      '251-275':['0-3','4-6','7-9','10-11','12+'],'276-300':['0-3','4-6','7-9','10-12','13+'],
      '301-325':['0-3','4-7','8-10','11-13','14+'],'326-350':['0-3','4-7','8-11','12-14','15+'],
      '351-375':['0-4','5-8','9-12','13-15','16+'],'376-400':['0-4','5-8','9-13','14-17','18+'],
      '401-425':['0-4','5-9','10-14','15-18','19+'],'426-450':['0-4','5-9','10-14','15-19','20+'],
      '451-475':['0-5','6-10','11-16','17-20','21+'],'476-500':['0-5','6-10','11-16','17-21','22+'],
      '501+':['0-5','6-11','12-17','18-21','22+']
    },
    C5: {
      '101-125':['0-1','2-3','4-5','6','7+'],'126-150':['0-2','3-4','5-6','7-8','9+'],
      '151-175':['0-2','3-4','5-6','7-9','10+'],'176-200':['0-2','3-4','5-7','8-10','11+'],
      '201-225':['0-3','4-6','7-9','10-12','13+'],'226-250':['0-3','4-6','7-10','11-13','14+'],
      '251-275':['0-3','4-7','8-11','12-15','16+'],'276-300':['0-4','5-8','9-12','13-16','17+'],
      '301-325':['0-4','5-8','9-13','14-17','18+'],'326-350':['0-4','5-9','10-14','15-19','20+'],
      '351-375':['0-5','6-10','11-15','16-20','21+'],'376-400':['0-5','6-11','12-16','17-22','23+'],
      '401-425':['0-5','6-11','12-17','18-23','24+'],'426-450':['0-6','7-12','13-18','19-25','26+'],
      '451-475':['0-6','7-13','14-19','20-26','27+'],'476-500':['0-6','7-13','14-20','21-27','28+'],
      '501+':['0-6','7-13','14-20','21-28','29+']
    }
  },
  sec5: {
    C4: {
      '101-125':['0','1','2','3','4+'],'126-150':['0-1','2','3','4','5+'],
      '151-175':['0-1','2-3','4','5','6+'],'176-200':['0-1','2-3','4','5-6','7+'],
      '201-225':['0-2','3-4','5-6','7','8+'],'226-250':['0-2','3-4','5-6','7-8','9+'],
      '251-275':['0-2','3-4','5-7','8-9','10+'],'276-300':['0-2','3-5','6-8','9-10','11+'],
      '301-325':['0-2','3-5','6-9','10-11','12+'],'326-350':['0-3','4-6','7-9','10-12','13+'],
      '351-375':['0-3','4-7','8-10','11-12','13+'],'376-400':['0-3','4-7','8-11','12-13','14+'],
      '401-425':['0-3','4-8','9-11','12-14','15+'],'426-450':['0-3','4-8','9-12','13-15','16+'],
      '451-475':['0-4','5-8','9-13','14-16','17+'],'476-500':['0-4','5-9','10-14','15-17','18+'],
      '501+':['0-4','5-9','10-14','15-17','18+']
    },
    C5: {
      '101-125':['0-1','2','3','4','5+'],'126-150':['0-1','2-3','4','5','6+'],
      '151-175':['0-1','2-3','4','5-6','7+'],'176-200':['0-2','3-4','5','6-7','8+'],
      '201-225':['0-2','3-4','5-6','7-8','9+'],'226-250':['0-2','3-4','5-6','7-8','9+'],
      '251-275':['0-2','3-5','6-7','8-9','10+'],'276-300':['0-2','3-5','6-8','9-10','11+'],
      '301-325':['0-3','4-5','6-9','10-11','12+'],'326-350':['0-3','4-6','7-10','11-12','13+'],
      '351-375':['0-3','4-6','7-10','11-13','14+'],'376-400':['0-3','4-7','8-11','12-14','15+'],
      '401-425':['0-3','4-8','9-12','13-15','16+'],'426-450':['0-4','5-8','9-12','13-16','17+'],
      '451-475':['0-4','5-9','10-13','14-17','18+'],'476-500':['0-4','5-9','10-14','15-18','19+'],
      '501+':['0-4','5-9','10-14','15-18','19+']
    }
  }
};

const COTE_HEADERS = ['A','B','C','D','E'];
const COTE_COLORS_BG = {
  A: 'bg-green-50', B: 'bg-blue-50', C: 'bg-yellow-50', D: 'bg-orange-50', E: 'bg-red-50'
};
const COTE_COLORS_HEADER = {
  A: 'bg-green-600 text-white', B: 'bg-blue-600 text-white', C: 'bg-yellow-500 text-white',
  D: 'bg-orange-500 text-white', E: 'bg-red-600 text-white'
};

function getMatchingRange(wordCount) {
  if (!wordCount) return null;
  const n = parseInt(wordCount);
  if (isNaN(n)) return null;
  if (n > 500) return '501+';
  for (const range of WORD_RANGES) {
    if (range === '501+') continue;
    const [lo, hi] = range.split('-').map(Number);
    if (n >= lo && n <= hi) return range;
  }
  return null;
}

const BaremeTable = ({ data, title, highlightRange }) => (
  <div className="space-y-1">
    <p className="text-xs font-semibold text-orange-700">{title}</p>
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="w-full text-[11px]">
        <thead>
          <tr>
            <th className="bg-gray-100 text-gray-600 px-2 py-1.5 text-left font-semibold sticky left-0 z-10 min-w-[70px]">Mots</th>
            {COTE_HEADERS.map(c => (
              <th key={c} className={`px-2 py-1.5 text-center font-bold ${COTE_COLORS_HEADER[c]} min-w-[52px]`}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {WORD_RANGES.map((range) => {
            const isHighlighted = range === highlightRange;
            return (
              <tr key={range} className={isHighlighted ? 'ring-2 ring-orange-400 ring-inset bg-orange-50 font-bold' : 'even:bg-gray-50/50'}>
                <td className={`px-2 py-1 font-medium text-gray-700 sticky left-0 z-10 ${isHighlighted ? 'bg-orange-50' : 'bg-white even:bg-gray-50/50'}`}>{range}</td>
                {data[range].map((val, i) => (
                  <td key={i} className={`px-2 py-1 text-center ${isHighlighted ? 'bg-orange-50' : COTE_COLORS_BG[COTE_HEADERS[i]]}`}>{val}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  </div>
);

const TextCorrectionModal = ({ open, onClose, onSubmit, onSubmitBatch, apiUrl }) => {
  const [niveau, setNiveau] = useState('');
  const [genreTexte, setGenreTexte] = useState('');
  const [genreAutre, setGenreAutre] = useState('');
  const [consigneEcriture, setConsigneEcriture] = useState('');
  const [profilScripteur, setProfilScripteur] = useState(false);
  const [detectionPlagiat, setDetectionPlagiat] = useState(false);
  const [nomEleve, setNomEleve] = useState('');
  const [groupeClasse, setGroupeClasse] = useState('');
  const [showProgress, setShowProgress] = useState(false);
  const [consignesBank, setConsignesBank] = useState([]);
  const [selectedConsigneId, setSelectedConsigneId] = useState('');
  const [consigneSaveStatus, setConsigneSaveStatus] = useState('');
  const [criteresActifs, setCriteresActifs] = useState({ C1: true, C2: true, C3: true, C4: true, C5: true });
  const [ponderations, setPonderations] = useState({ C1: 25, C2: 20, C3: 10, C4: 25, C5: 20 });
  const [totalPoints, setTotalPoints] = useState('');
  const [descripteursC1, setDescripteursC1] = useState('');
  const [nombreMots, setNombreMots] = useState('');
  const [texteEleve, setTexteEleve] = useState('');
  const [consignesSupp, setConsignesSupp] = useState('');
  const [tableauFormat, setTableauFormat] = useState('numero'); // 'numero' ou 'type'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [inputMode, setInputMode] = useState('text'); // 'text', 'file' ou 'batch'
  const [batchFiles, setBatchFiles] = useState([]);

  useEffect(() => {
    if (!niveau) return;
    const key = niveau === 'sec5' ? 'sec5' : 'sec1-4';
    setPonderations({ ...PONDERATIONS[key] });
  }, [niveau]);

  const loadConsignes = useCallback(async () => {
    try {
      const res = await axios.get(`${apiUrl}/consignes`);
      setConsignesBank(res.data.consignes || []);
    } catch (e) { /* non connecté — banque indisponible */ }
  }, [apiUrl]);

  const [knownStudents, setKnownStudents] = useState([]);
  const loadKnownStudents = useCallback(async () => {
    try {
      const res = await axios.get(`${apiUrl}/student-profiles`);
      setKnownStudents(res.data.students || []);
    } catch (e) { /* non connecté */ }
  }, [apiUrl]);

  useEffect(() => {
    if (open) {
      loadConsignes();
      loadKnownStudents();
    }
  }, [open, loadConsignes, loadKnownStudents]);

  const handleNomEleveChange = (val) => {
    setNomEleve(val);
    const match = knownStudents.find(s => s.name.toLowerCase() === val.trim().toLowerCase());
    if (match && match.groupe) setGroupeClasse(match.groupe);
  };

  const saveConsigne = async () => {
    if (!consigneEcriture.trim()) return;
    setConsigneSaveStatus('saving');
    try {
      const titre = consigneEcriture.trim().split(/\s+/).slice(0, 8).join(' ');
      const res = await axios.post(`${apiUrl}/consignes`, { titre, texte: consigneEcriture.trim() });
      setConsignesBank(prev => [res.data.consigne, ...prev]);
      setConsigneSaveStatus('saved');
      setTimeout(() => setConsigneSaveStatus(''), 2500);
    } catch (e) {
      setConsigneSaveStatus('error');
      setTimeout(() => setConsigneSaveStatus(''), 2500);
    }
  };

  const loadConsigneFromBank = (id) => {
    setSelectedConsigneId(id);
    const c = consignesBank.find(x => x.id === id);
    if (c) setConsigneEcriture(c.texte);
  };

  const deleteSelectedConsigne = async () => {
    if (!selectedConsigneId) return;
    try {
      await axios.delete(`${apiUrl}/consignes/${selectedConsigneId}`);
      setConsignesBank(prev => prev.filter(c => c.id !== selectedConsigneId));
      setSelectedConsigneId('');
    } catch (e) { /* ignore */ }
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp'
    ];
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExts = ['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'webp'];

    if (!allowedExts.includes(ext)) {
      setUploadError('Format non supporté. Formats acceptés : PDF, Word (.docx), Texte (.txt), Images (PNG, JPG)');
      return;
    }

    setIsUploading(true);
    setUploadError('');
    setUploadedFile({ name: file.name, size: file.size });

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${apiUrl}/upload-file`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Erreur lors du traitement du fichier');
      }

      const data = await response.json();
      setTexteEleve(data.extracted_text || '');
      setUploadedFile({ name: file.name, size: file.size, success: true });
    } catch (err) {
      setUploadError(err.message || 'Erreur lors du traitement du fichier');
      setUploadedFile(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFileUpload(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileUpload(file);
  };

  const removeFile = () => {
    setUploadedFile(null);
    setTexteEleve('');
    setUploadError('');
  };

  const highlightRange = useMemo(() => {
    // Priorité: nombre de mots du texte collé, sinon champ manuel
    const wordCount = texteEleve.trim() ? texteEleve.trim().split(/\s+/).length : parseInt(nombreMots);
    return getMatchingRange(wordCount || nombreMots);
  }, [nombreMots, texteEleve]);

  const actualWordCount = useMemo(() => {
    if (texteEleve.trim()) return texteEleve.trim().split(/\s+/).length;
    return parseInt(nombreMots) || null;
  }, [texteEleve, nombreMots]);

  const toggleCritere = (crit) => {
    setCriteresActifs(prev => ({ ...prev, [crit]: !prev[crit] }));
  };

  const updatePonderation = (crit, val) => {
    setPonderations(prev => ({ ...prev, [crit]: parseInt(val) || 0 }));
  };

  const totalPond = Object.entries(ponderations)
    .filter(([k]) => criteresActifs[k])
    .reduce((sum, [, v]) => sum + v, 0);

  const buildCorrectionMessage = (texteParam, eleveNameParam) => {
    const criteresTexte = Object.entries(criteresActifs)
      .filter(([, v]) => v)
      .map(([k]) => `${k} - ${CRITERES_LABELS[k]} (${ponderations[k]}%)`)
      .join('\n   ');

    const wordCount = texteParam.trim() ? texteParam.trim().split(/\s+/).length : (parseInt(nombreMots) || null);
    const range = getMatchingRange(wordCount || nombreMots);

    let message = `Corrige ce texte d'élève selon le protocole MEQ et attribue les cotes et la note finale automatiquement.\n\n`;
    message += `**Niveau:** ${niveau ? niveau.replace('sec', 'Secondaire ') : 'Non précisé'}\n`;
    if (eleveNameParam && eleveNameParam.trim()) {
      message += `**Élève:** ${eleveNameParam.trim()}\n`;
    }
    if (groupeClasse.trim()) {
      message += `**Groupe:** ${groupeClasse.trim()}\n`;
    }

    // Genre de texte + grille spécifique
    if (genreTexte) {
      const genreLabel = genreTexte === 'autre'
        ? (genreAutre.trim() || 'Autre')
        : GENRES_TEXTE.find(g => g.value === genreTexte)?.label;
      message += `**Genre de texte:** ${genreLabel}\n`;
      if (GRILLES_GENRE[genreTexte]) {
        message += `\n**GRILLE D'ÉVALUATION SPÉCIFIQUE AU GENRE (inspirée des grilles Tshakapesh) — utilise ces descripteurs pour évaluer C1 (adaptation) et C2 (cohérence):**\n${GRILLES_GENRE[genreTexte]}\n\n`;
      }
    }

    // Consigne d'écriture
    if (consigneEcriture.trim()) {
      message += `**Consigne d'écriture donnée aux élèves (pour l'ensemble de l'œuvre):** ${consigneEcriture.trim()}\n`;
      message += `**INSTRUCTION:** Évalue le critère C1 (adaptation à la situation de communication) en vérifiant si le texte RESPECTE cette consigne (sujet, genre, longueur, contraintes).\n`;
    }

    message += `**Critères évalués et pondération:**\n   ${criteresTexte}\n`;
    message += `**Nombre total de points:** ${totalPoints || 'Non précisé'}\n`;

    if (descripteursC1.trim()) {
      message += `**Descripteurs du critère 1 (Adaptation):** ${descripteursC1}\n`;
    }

    message += `**Nombre de mots du texte:** ${wordCount || 'Non précisé'}\n`;

    // Barèmes officiels basés sur le niveau et le nombre de mots
    if (niveau && BAREMES_COMPLETS[niveau] && range) {
      const c4 = BAREMES_COMPLETS[niveau].C4[range];
      const c5 = BAREMES_COMPLETS[niveau].C5[range];
      message += `\n**BARÈME OFFICIEL À UTILISER (${range} mots, ${niveau.replace('sec','Sec ')}):**\n`;
      message += `  C4 (Syntaxe/Ponctuation): A=${c4[0]} err, B=${c4[1]} err, C=${c4[2]} err, D=${c4[3]} err, E=${c4[4]} err\n`;
      message += `  C5 (Orthographe):         A=${c5[0]} err, B=${c5[1]} err, C=${c5[2]} err, D=${c5[3]} err, E=${c5[4]} err\n`;
      message += `\n**INSTRUCTION:** Compte les erreurs C4 et C5 selon les règles MEQ, consulte ce barème, attribue la cote (A/B/C/D/E) et calcule la note finale.\n`;
    }

    if (consignesSupp.trim()) {
      message += `\n**Consignes supplémentaires:** ${consignesSupp}\n`;
    }

    const formatLabels = {
      'numero': "Option 1 — Par NUMÉRO (ordre d'apparition dans le texte)",
      'type': "Option 2 — Par TYPE de faute (regrouper les erreurs par catégorie: S, P, U, G, V, C1, C2)",
      'les_deux': "Option 3 — LES DEUX: d'abord le tableau par numéro (option 1), puis le même tableau réorganisé par type de faute (option 2)"
    };
    message += `\n**Format du tableau de corrections:** ${formatLabels[tableauFormat]}\n`;

    // Profil scripteur (PS)
    if (profilScripteur) {
      message += `\n[PROFIL_SCRIPTEUR:${niveau || 'sec3'}]\n`;
      message += `**PROFIL SCRIPTEUR (PS) DEMANDÉ:** Produis le profil de scripteur complet de l'élève après la correction.\n`;
    }

    // Détection de plagiat
    if (detectionPlagiat) {
      message += `\n[DETECTION_PLAGIAT]\n`;
      message += `**DÉTECTION DE PLAGIAT DEMANDÉE:** Analyse le texte pour détecter d'éventuels passages plagiés.\n`;
    }

    message += `\n---\n**TEXTE DE L'ÉLÈVE À CORRIGER:**\n\n${texteParam}`;
    return message;
  };

  const handleSubmit = () => {
    onSubmit(buildCorrectionMessage(texteEleve, nomEleve));
    onClose();
  };

  const handleBatchSubmit = () => {
    const ready = batchFiles.filter(f => f.status === 'ready' && f.text.trim());
    if (ready.length === 0) return;
    onSubmitBatch(ready.map(f => ({
      name: f.studentName,
      message: buildCorrectionMessage(f.text, f.studentName)
    })));
    onClose();
  };

  const handleBatchFilesSelect = async (fileList) => {
    const allowedExts = ['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'bmp', 'webp'];
    const files = Array.from(fileList || []).slice(0, 15);
    const newEntries = files.map((f, idx) => {
      const ok = allowedExts.includes(f.name.split('.').pop().toLowerCase());
      return {
        id: `${Date.now()}_${idx}`,
        file: f,
        fileName: f.name,
        studentName: f.name.replace(/\.[^.]+$/, '').replace(/[_\-.]+/g, ' ').replace(/\s+/g, ' ').trim(),
        text: '',
        status: ok ? 'pending' : 'error',
        error: ok ? '' : 'Format non supporté'
      };
    });
    setBatchFiles(prev => [...prev, ...newEntries]);
    for (const entry of newEntries) {
      if (entry.status === 'error') continue;
      setBatchFiles(prev => prev.map(x => x.id === entry.id ? { ...x, status: 'extracting' } : x));
      try {
        const formData = new FormData();
        formData.append('file', entry.file);
        const response = await fetch(`${apiUrl}/upload-file`, { method: 'POST', body: formData });
        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Erreur lors du traitement');
        }
        const data = await response.json();
        const txt = (data.extracted_text || '').trim();
        setBatchFiles(prev => prev.map(x => x.id === entry.id
          ? { ...x, text: txt, status: txt ? 'ready' : 'error', error: txt ? '' : 'Aucun texte extrait' }
          : x));
      } catch (err) {
        setBatchFiles(prev => prev.map(x => x.id === entry.id ? { ...x, status: 'error', error: err.message } : x));
      }
    }
  };

  const updateBatchEntry = (id, field, value) => {
    setBatchFiles(prev => prev.map(x => x.id === id ? { ...x, [field]: value } : x));
  };

  const removeBatchEntry = (id) => {
    setBatchFiles(prev => prev.filter(x => x.id !== id));
  };

  const batchReadyCount = batchFiles.filter(f => f.status === 'ready' && f.text.trim()).length;
  const batchBusy = batchFiles.some(f => f.status === 'extracting' || f.status === 'pending');

  const canSubmit = inputMode === 'batch' ? (batchReadyCount > 0 && !batchBusy) : texteEleve.trim().length > 0;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto" data-testid="text-correction-modal">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-orange-600"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
            Corriger un texte d'élève
          </DialogTitle>
          <p className="text-sm text-gray-500">Protocole de correction conforme aux grilles d'évaluation du MEQ</p>
        </DialogHeader>

        <div className="space-y-5 mt-2">
          {/* Niveau scolaire + Genre de texte */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label className="font-semibold text-gray-700">Niveau scolaire</Label>
              <Select value={niveau} onValueChange={setNiveau}>
                <SelectTrigger data-testid="niveau-select" className="border-gray-300">
                  <SelectValue placeholder="Choisir le niveau..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sec1">Secondaire 1</SelectItem>
                  <SelectItem value="sec2">Secondaire 2</SelectItem>
                  <SelectItem value="sec3">Secondaire 3</SelectItem>
                  <SelectItem value="sec4">Secondaire 4</SelectItem>
                  <SelectItem value="sec5">Secondaire 5</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="font-semibold text-gray-700">Genre de texte</Label>
              <Select value={genreTexte} onValueChange={setGenreTexte}>
                <SelectTrigger data-testid="genre-select" className="border-gray-300">
                  <SelectValue placeholder="Choisir le genre..." />
                </SelectTrigger>
                <SelectContent>
                  {GENRES_TEXTE.map(g => (
                    <SelectItem key={g.value} value={g.value}>{g.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {genreTexte && genreTexte !== 'autre' && (
                <p className="text-[10px] text-orange-600">Grille d'évaluation spécifique au genre appliquée (inspirée des grilles Tshakapesh)</p>
              )}
            </div>
          </div>

          {/* Champ libre si genre = autre */}
          {genreTexte === 'autre' && (
            <div className="space-y-1.5">
              <Label className="font-semibold text-gray-700">Précisez le genre de texte</Label>
              <Input
                data-testid="genre-autre-input"
                type="text"
                placeholder="Ex: haïku, chanson, discours, texte d'opinion..."
                value={genreAutre}
                onChange={(e) => setGenreAutre(e.target.value)}
                className="border-gray-300"
              />
            </div>
          )}

          {/* Nom de l'élève + Groupe + Suivi */}
          <div className="flex items-end gap-3">
            <div className="space-y-1.5 flex-1">
              <Label className="font-semibold text-gray-700">
                Nom de l'élève <span className="text-xs text-gray-400 font-normal">(optionnel — active le suivi de progression)</span>
              </Label>
              <Input
                data-testid="nom-eleve-input"
                type="text"
                placeholder="Ex: Émile Tremblay"
                value={nomEleve}
                onChange={(e) => handleNomEleveChange(e.target.value)}
                className="border-gray-300"
                list="known-students-datalist"
              />
              <datalist id="known-students-datalist">
                {knownStudents.map(s => (
                  <option key={s.name} value={s.name} />
                ))}
              </datalist>
            </div>
            <div className="space-y-1.5 w-36">
              <Label className="font-semibold text-gray-700">Groupe-classe</Label>
              <Input
                data-testid="groupe-classe-input"
                type="text"
                placeholder="Ex: 32"
                value={groupeClasse}
                onChange={(e) => setGroupeClasse(e.target.value)}
                className="border-gray-300"
              />
            </div>
            <Button
              type="button"
              variant="outline"
              onClick={() => setShowProgress(true)}
              className="border-orange-300 text-orange-700 hover:bg-orange-50"
              data-testid="open-progress-btn"
            >
              📈 Suivi des élèves
            </Button>
          </div>
          {nomEleve.trim() && !profilScripteur && (
            <p className="text-xs text-amber-600 -mt-3">💡 Cochez « Profil scripteur (PS) » ci-dessous pour que cette correction s'ajoute au suivi de {nomEleve.trim()}.</p>
          )}

          {/* Consigne d'écriture */}
          <div className="space-y-1.5">
            <Label className="font-semibold text-gray-700">
              Consigne d'écriture <span className="text-xs text-gray-400 font-normal">(pour l'ensemble de l'œuvre, optionnel)</span>
            </Label>
            <p className="text-xs text-gray-500">La consigne donnée aux élèves. Étienne vérifiera si le texte la respecte (critère C1).</p>

            {/* Banque de consignes */}
            <div className="flex gap-2 items-center">
              <Select value={selectedConsigneId} onValueChange={loadConsigneFromBank}>
                <SelectTrigger data-testid="consignes-bank-select" className="border-gray-300 flex-1 h-9 text-sm">
                  <SelectValue placeholder={consignesBank.length > 0 ? `📂 Charger une consigne sauvegardée (${consignesBank.length})...` : '📂 Aucune consigne sauvegardée'} />
                </SelectTrigger>
                <SelectContent>
                  {consignesBank.map(c => (
                    <SelectItem key={c.id} value={c.id}>{c.titre}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={saveConsigne}
                disabled={!consigneEcriture.trim() || consigneSaveStatus === 'saving'}
                className="h-9 whitespace-nowrap"
                data-testid="save-consigne-btn"
              >
                {consigneSaveStatus === 'saved' ? '✅ Sauvegardée' : consigneSaveStatus === 'error' ? '❌ Erreur' : '💾 Sauvegarder'}
              </Button>
              {selectedConsigneId && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={deleteSelectedConsigne}
                  className="h-9 text-red-500 hover:text-red-600 hover:bg-red-50"
                  title="Supprimer cette consigne de la banque"
                  data-testid="delete-consigne-btn"
                >
                  🗑
                </Button>
              )}
            </div>

            <Textarea
              data-testid="consigne-ecriture-input"
              placeholder="Ex: Rédige un récit fantastique de 300 à 400 mots se déroulant dans ton école. Ton récit doit contenir un élément surnaturel et maintenir le doute chez le lecteur..."
              value={consigneEcriture}
              onChange={(e) => setConsigneEcriture(e.target.value)}
              rows={2}
              className="border-gray-300 text-sm"
            />
          </div>

          <Separator />

          {/* Critères et pondérations */}
          <div className="space-y-2">
            <Label className="font-semibold text-gray-700">Critères évalués et pondération (%)</Label>
            <p className="text-xs text-gray-500">Cochez les critères et ajustez les pondérations selon vos besoins</p>
            <div className="space-y-2 bg-gray-50 rounded-lg p-3">
              {Object.entries(CRITERES_LABELS).map(([key, label]) => (
                <div key={key} className="flex items-center gap-3" data-testid={`critere-${key}`}>
                  <Checkbox
                    checked={criteresActifs[key]}
                    onCheckedChange={() => toggleCritere(key)}
                    id={`critere-${key}`}
                  />
                  <label htmlFor={`critere-${key}`} className="text-sm text-gray-700 flex-1 cursor-pointer">
                    <span className="font-medium text-orange-700">{key}</span> - {label}
                  </label>
                  <Input
                    type="number"
                    min="0"
                    max="100"
                    value={ponderations[key]}
                    onChange={(e) => updatePonderation(key, e.target.value)}
                    disabled={!criteresActifs[key]}
                    className="w-16 h-8 text-center text-sm"
                  />
                  <span className="text-xs text-gray-500">%</span>
                </div>
              ))}
              <div className={`text-xs font-medium text-right mt-1 ${totalPond === 100 ? 'text-green-600' : 'text-red-500'}`}>
                Total: {totalPond}% {totalPond !== 100 && '(doit = 100%)'}
              </div>
            </div>
          </div>

          {/* Points totaux + Nombre de mots */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <Label className="font-semibold text-gray-700">Points totaux</Label>
              <Input
                data-testid="total-points-input"
                type="text"
                placeholder="Ex: /40, /50, /100"
                value={totalPoints}
                onChange={(e) => setTotalPoints(e.target.value)}
                className="border-gray-300"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="font-semibold text-gray-700">Nombre de mots</Label>
              <Input
                data-testid="word-count-input"
                type="text"
                placeholder="Auto-calculé ou entrer manuellement"
                value={actualWordCount || nombreMots}
                onChange={(e) => setNombreMots(e.target.value)}
                readOnly={texteEleve.trim().length > 0}
                className={`border-gray-300 ${texteEleve.trim().length > 0 ? 'bg-gray-50 text-gray-600' : ''}`}
              />
              {highlightRange && (
                <p className="text-xs text-orange-600 font-medium">Plage correspondante: {highlightRange} mots</p>
              )}
              {texteEleve.trim().length > 0 && (
                <p className="text-[10px] text-gray-400">Calculé automatiquement depuis le texte</p>
              )}
            </div>
          </div>

          <Separator />

          {/* Descripteurs C1 */}
          <div className="space-y-1.5">
            <Label className="font-semibold text-gray-700">
              Descripteurs du critère 1 <span className="text-xs text-gray-400 font-normal">(optionnel)</span>
            </Label>
            <p className="text-xs text-gray-500">Ce critère est subjectif. Précisez les éléments que vous évaluez pour cette tâche.</p>
            <Textarea
              data-testid="descripteurs-c1-input"
              placeholder="Ex: Respect du sujet, pertinence des arguments, registre de langue courant, texte justificatif de 250 mots min..."
              value={descripteursC1}
              onChange={(e) => setDescripteursC1(e.target.value)}
              rows={2}
              className="border-gray-300 text-sm"
            />
          </div>

          {/* Barèmes officiels C4/C5 - TABLEAUX COMPLETS EN LECTURE SEULE */}
          <div className="space-y-2">
            <Label className="font-semibold text-gray-700">
              Barèmes officiels C4 et C5 <span className="text-xs text-gray-400 font-normal">(repères CS Laval / MEQ)</span>
            </Label>
            {!niveau ? (
              <p className="text-sm text-gray-400 italic bg-gray-50 rounded-lg p-3">Sélectionnez un niveau scolaire pour afficher les barèmes officiels.</p>
            ) : (
              <div className="space-y-3" data-testid="baremes-tables">
                <BaremeTable
                  data={BAREMES_COMPLETS[niveau].C4}
                  title="C4 - Construction des phrases (syntaxe) et ponctuation"
                  highlightRange={highlightRange}
                />
                <BaremeTable
                  data={BAREMES_COMPLETS[niveau].C5}
                  title="C5 - Orthographe d'usage et grammaticale"
                  highlightRange={highlightRange}
                />
                <p className="text-[10px] text-gray-400">Source: Repères pour l'attribution d'une cote, critères 4 et 5, sec. 1 à 5 (CS Laval / MEQ).</p>
              </div>
            )}
          </div>

          <Separator />

          {/* Format du tableau de corrections */}
          <div className="space-y-1.5">
            <Label className="font-semibold text-gray-700">Format du tableau de corrections</Label>
            <div className="flex gap-3 flex-wrap">
              <label 
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border cursor-pointer transition-all ${
                  tableauFormat === 'numero' 
                    ? 'border-orange-400 bg-orange-50 ring-1 ring-orange-300' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                data-testid="tableau-format-numero"
              >
                <input
                  type="radio"
                  name="tableauFormat"
                  value="numero"
                  checked={tableauFormat === 'numero'}
                  onChange={(e) => setTableauFormat(e.target.value)}
                  className="text-orange-500"
                />
                <div>
                  <span className="text-sm font-medium text-gray-800">Par numéro</span>
                  <p className="text-xs text-gray-500">Ordre d'apparition dans le texte</p>
                </div>
              </label>
              <label 
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border cursor-pointer transition-all ${
                  tableauFormat === 'type' 
                    ? 'border-orange-400 bg-orange-50 ring-1 ring-orange-300' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                data-testid="tableau-format-type"
              >
                <input
                  type="radio"
                  name="tableauFormat"
                  value="type"
                  checked={tableauFormat === 'type'}
                  onChange={(e) => setTableauFormat(e.target.value)}
                  className="text-orange-500"
                />
                <div>
                  <span className="text-sm font-medium text-gray-800">Par type</span>
                  <p className="text-xs text-gray-500">Regroupées par catégorie</p>
                </div>
              </label>
              <label 
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border cursor-pointer transition-all ${
                  tableauFormat === 'les_deux' 
                    ? 'border-orange-400 bg-orange-50 ring-1 ring-orange-300' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                data-testid="tableau-format-les-deux"
              >
                <input
                  type="radio"
                  name="tableauFormat"
                  value="les_deux"
                  checked={tableauFormat === 'les_deux'}
                  onChange={(e) => setTableauFormat(e.target.value)}
                  className="text-orange-500"
                />
                <div>
                  <span className="text-sm font-medium text-gray-800">Les deux</span>
                  <p className="text-xs text-gray-500">Par numéro + par type ensemble</p>
                </div>
              </label>
            </div>
          </div>

          {/* Options avancées: Profil scripteur + Détection de plagiat */}
          <div className="space-y-2">
            <Label className="font-semibold text-gray-700">Options d'analyse avancées</Label>
            <div className="space-y-2.5 bg-gradient-to-r from-orange-50/60 to-amber-50/60 border border-orange-100 rounded-lg p-3">
              <div className="flex items-start gap-3" data-testid="option-profil-scripteur">
                <Checkbox
                  id="ps-checkbox"
                  checked={profilScripteur}
                  onCheckedChange={(v) => setProfilScripteur(!!v)}
                  className="mt-0.5"
                />
                <label htmlFor="ps-checkbox" className="cursor-pointer flex-1">
                  <span className="text-sm font-medium text-gray-800">Profil scripteur (PS)</span>
                  <p className="text-xs text-gray-500">Produit le profil diagnostique de l'élève : chaque erreur classée par code (C, V, S, P, U, G) selon la taxonomie du niveau, avec bilan et recommandations.</p>
                  {profilScripteur && !niveau && (
                    <p className="text-[10px] text-red-500 mt-0.5">Sélectionnez le niveau scolaire pour utiliser la bonne taxonomie.</p>
                  )}
                </label>
              </div>
              <div className="flex items-start gap-3" data-testid="option-detection-plagiat">
                <Checkbox
                  id="plagiat-checkbox"
                  checked={detectionPlagiat}
                  onCheckedChange={(v) => setDetectionPlagiat(!!v)}
                  className="mt-0.5"
                />
                <label htmlFor="plagiat-checkbox" className="cursor-pointer flex-1">
                  <span className="text-sm font-medium text-gray-800">Détection de plagiat</span>
                  <p className="text-xs text-gray-500">Recherche sur le web les passages potentiellement copiés (sources + URL) et analyse stylistique. Les passages suspects sont soulignés en gris dans le texte annoté.</p>
                  {detectionPlagiat && (
                    <p className="text-xs text-amber-600 mt-0.5">⚠️ Analyse plus longue (recherche web) et consomme un peu plus de quota.</p>
                  )}
                </label>
              </div>
            </div>
          </div>

          {/* Consignes supplémentaires */}
          <div className="space-y-1.5">
            <Label className="font-semibold text-gray-700">
              Consignes supplémentaires <span className="text-xs text-gray-400 font-normal">(optionnel)</span>
            </Label>
            <Textarea
              data-testid="consignes-input"
              placeholder="Ex: Sois indulgent avec l'orthographe, c'est un brouillon. / Focus sur les accords du participe passé / Ne pas pénaliser les anglicismes..."
              value={consignesSupp}
              onChange={(e) => setConsignesSupp(e.target.value)}
              rows={2}
              className="border-gray-300 text-sm"
            />
          </div>

          {/* Texte de l'élève - deux modes */}
          <div className="space-y-2">
            <Label className="font-semibold text-gray-700">
              Texte de l'élève <span className="text-red-500">*</span>
            </Label>

            {/* Sélection du mode */}
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => { setInputMode('text'); removeFile(); }}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium border transition-all ${
                  inputMode === 'text'
                    ? 'bg-orange-50 border-orange-300 text-orange-700'
                    : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                }`}
                data-testid="input-mode-text"
              >
                <span className="flex items-center justify-center gap-1.5">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
                  Copier-coller
                </span>
              </button>
              <button
                type="button"
                onClick={() => setInputMode('file')}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium border transition-all ${
                  inputMode === 'file'
                    ? 'bg-orange-50 border-orange-300 text-orange-700'
                    : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                }`}
                data-testid="input-mode-file"
              >
                <span className="flex items-center justify-center gap-1.5">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>
                  Joindre un fichier
                </span>
              </button>
              <button
                type="button"
                onClick={() => setInputMode('batch')}
                className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium border transition-all ${
                  inputMode === 'batch'
                    ? 'bg-orange-50 border-orange-300 text-orange-700'
                    : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                }`}
                data-testid="input-mode-batch"
              >
                <span className="flex items-center justify-center gap-1.5">
                  <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 7h-3a2 2 0 0 1-2-2V2"/><path d="M9 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h7l5 5v9a2 2 0 0 1-2 2Z"/><path d="M3 7.6v12.8A1.6 1.6 0 0 0 4.6 22h9.8"/></svg>
                  Lot de fichiers
                </span>
              </button>
            </div>

            {/* Mode copier-coller */}
            {inputMode === 'text' && (
              <>
                <Textarea
                  data-testid="texte-eleve-input"
                  placeholder="Collez le texte de l'élève ici..."
                  value={texteEleve}
                  onChange={(e) => setTexteEleve(e.target.value)}
                  rows={6}
                  className="border-gray-300"
                />
                {texteEleve.trim() && (
                  <p className="text-xs text-gray-500">
                    ~{texteEleve.trim().split(/\s+/).length} mots
                  </p>
                )}
              </>
            )}

            {/* Mode fichier joint */}
            {inputMode === 'file' && (
              <div className="space-y-2">
                {!uploadedFile && !isUploading && (
                  <div
                    onDrop={handleDrop}
                    onDragOver={(e) => e.preventDefault()}
                    className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-orange-400 hover:bg-orange-50/30 transition-colors cursor-pointer"
                    onClick={() => document.getElementById('file-input-correction')?.click()}
                    data-testid="file-drop-zone"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mx-auto text-gray-400 mb-2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>
                    </svg>
                    <p className="text-sm text-gray-600 font-medium">Glissez un fichier ici ou cliquez pour parcourir</p>
                    <p className="text-xs text-gray-400 mt-1">PDF, Word (.docx), Texte (.txt), Images (PNG, JPG)</p>
                    <p className="text-xs text-orange-500 mt-1 font-medium">✍️ Textes manuscrits acceptés : photo ou scan d'un texte écrit à la main (conversion automatique)</p>
                    <input
                      id="file-input-correction"
                      type="file"
                      className="hidden"
                      accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.webp"
                      onChange={handleFileInputChange}
                    />
                  </div>
                )}

                {isUploading && (
                  <div className="border border-orange-200 bg-orange-50 rounded-lg p-4 text-center">
                    <div className="animate-spin h-6 w-6 border-2 border-orange-500 border-t-transparent rounded-full mx-auto mb-2"></div>
                    <p className="text-sm text-orange-700">Extraction du texte en cours...</p>
                    <p className="text-xs text-orange-500">{uploadedFile?.name}</p>
                  </div>
                )}

                {uploadedFile?.success && (
                  <div className="border border-green-200 bg-green-50 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-green-600"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                        <div>
                          <p className="text-sm font-medium text-green-800">{uploadedFile.name}</p>
                          <p className="text-xs text-green-600">Texte extrait avec succès</p>
                        </div>
                      </div>
                      <button onClick={removeFile} className="text-gray-400 hover:text-red-500 transition-colors" data-testid="remove-file-btn">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" x2="6" y1="6" y2="18"/><line x1="6" x2="18" y1="6" y2="18"/></svg>
                      </button>
                    </div>
                  </div>
                )}

                {uploadError && (
                  <div className="border border-red-200 bg-red-50 rounded-lg p-3">
                    <p className="text-sm text-red-700">{uploadError}</p>
                  </div>
                )}

                {/* Texte extrait (éditable) */}
                {texteEleve && inputMode === 'file' && (
                  <>
                    <div className="bg-amber-50 border border-amber-200 rounded-lg p-2.5" data-testid="ocr-verification-notice">
                      <p className="text-xs text-amber-700 font-medium">✍️ Vérifiez le texte converti ci-dessous AVANT de lancer la correction — surtout pour un texte manuscrit. Corrigez les mots mal déchiffrés ou marqués [illisible].</p>
                    </div>
                    <Label className="text-xs text-gray-500">Texte extrait (vous pouvez le modifier avant de corriger) :</Label>
                    <Textarea
                      data-testid="texte-eleve-input"
                      value={texteEleve}
                      onChange={(e) => setTexteEleve(e.target.value)}
                      rows={5}
                      className="border-gray-300 text-sm"
                    />
                    <p className="text-xs text-gray-500">~{texteEleve.trim().split(/\s+/).length} mots</p>
                  </>
                )}
              </div>
            )}

            {/* Mode lot de fichiers */}
            {inputMode === 'batch' && (
              <div className="space-y-2">
                <div className="bg-blue-50 border border-blue-100 rounded-lg p-2.5">
                  <p className="text-xs text-blue-700">📚 <strong>Correction en lot</strong> : téléversez plusieurs fichiers (un texte d'élève par fichier). Le nom de l'élève est déduit du nom du fichier — vous pouvez le modifier. Les paramètres ci-dessus (niveau, genre, consigne, PS, plagiat) s'appliquent à tous les textes. Les corrections apparaîtront une à une dans le chat.</p>
                </div>

                <div
                  onDrop={(e) => { e.preventDefault(); handleBatchFilesSelect(e.dataTransfer?.files); }}
                  onDragOver={(e) => e.preventDefault()}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-orange-400 hover:bg-orange-50/30 transition-colors cursor-pointer"
                  onClick={() => document.getElementById('batch-files-input')?.click()}
                  data-testid="batch-drop-zone"
                >
                  <p className="text-sm text-gray-600 font-medium">Glissez plusieurs fichiers ici ou cliquez pour parcourir</p>
                  <p className="text-xs text-gray-400 mt-1">PDF, Word, Texte, Images (max 15 fichiers) — manuscrits acceptés ✍️</p>
                  <input
                    id="batch-files-input"
                    type="file"
                    multiple
                    className="hidden"
                    accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.webp"
                    onChange={(e) => { handleBatchFilesSelect(e.target.files); e.target.value = ''; }}
                    data-testid="batch-files-input"
                  />
                </div>

                {batchFiles.length > 0 && (
                  <div className="space-y-1.5" data-testid="batch-files-list">
                    {batchFiles.map(f => (
                      <div key={f.id} className={`flex items-center gap-2 border rounded-lg p-2 ${f.status === 'error' ? 'border-red-200 bg-red-50' : f.status === 'ready' ? 'border-green-200 bg-green-50/60' : 'border-gray-200 bg-gray-50'}`}>
                        <span className="text-sm w-5 text-center">
                          {f.status === 'ready' ? '✅' : f.status === 'error' ? '❌' : (
                            <span className="inline-block w-3 h-3 border-2 border-orange-500 border-t-transparent rounded-full animate-spin"></span>
                          )}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-gray-500 truncate">{f.fileName}{f.status === 'ready' && ` — ~${f.text.trim().split(/\s+/).length} mots`}{f.error && ` — ${f.error}`}</p>
                          <input
                            type="text"
                            value={f.studentName}
                            onChange={(e) => updateBatchEntry(f.id, 'studentName', e.target.value)}
                            placeholder="Nom de l'élève"
                            className="w-full text-xs border border-gray-300 rounded px-2 py-1 mt-0.5 bg-white"
                            data-testid={`batch-student-name-${f.id}`}
                          />
                        </div>
                        <button onClick={() => removeBatchEntry(f.id)} className="text-gray-300 hover:text-red-500" data-testid={`batch-remove-${f.id}`}>✕</button>
                      </div>
                    ))}
                    <p className="text-xs text-gray-500">
                      {batchReadyCount} texte{batchReadyCount > 1 ? 's' : ''} prêt{batchReadyCount > 1 ? 's' : ''} sur {batchFiles.length}
                      {batchReadyCount > 1 && ' — une pause de ~8 s sera faite entre chaque correction pour respecter les limites d\'Étienne'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="outline" onClick={onClose} data-testid="correction-cancel-btn">
              Annuler
            </Button>
            <Button
              data-testid="correction-submit-btn"
              onClick={inputMode === 'batch' ? handleBatchSubmit : handleSubmit}
              disabled={!canSubmit}
              className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-medium px-6"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1.5"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
              {inputMode === 'batch'
                ? `Corriger les ${batchReadyCount} texte${batchReadyCount > 1 ? 's' : ''}`
                : 'Corriger le texte'}
            </Button>
          </div>
        </div>
      </DialogContent>
      <StudentProgressModal open={showProgress} onClose={() => setShowProgress(false)} apiUrl={apiUrl} />
    </Dialog>
  );
};

export default TextCorrectionModal;
