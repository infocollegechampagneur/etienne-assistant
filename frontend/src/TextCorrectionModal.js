import React, { useState, useEffect, useMemo } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Textarea } from './components/ui/textarea';
import { Checkbox } from './components/ui/checkbox';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Separator } from './components/ui/separator';

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

const TextCorrectionModal = ({ open, onClose, onSubmit }) => {
  const [niveau, setNiveau] = useState('');
  const [criteresActifs, setCriteresActifs] = useState({ C1: true, C2: true, C3: true, C4: true, C5: true });
  const [ponderations, setPonderations] = useState({ C1: 25, C2: 20, C3: 10, C4: 25, C5: 20 });
  const [totalPoints, setTotalPoints] = useState('');
  const [descripteursC1, setDescripteursC1] = useState('');
  const [nombreMots, setNombreMots] = useState('');
  const [texteEleve, setTexteEleve] = useState('');
  const [consignesSupp, setConsignesSupp] = useState('');

  useEffect(() => {
    if (!niveau) return;
    const key = niveau === 'sec5' ? 'sec5' : 'sec1-4';
    setPonderations({ ...PONDERATIONS[key] });
  }, [niveau]);

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

  const handleSubmit = () => {
    const criteresTexte = Object.entries(criteresActifs)
      .filter(([, v]) => v)
      .map(([k]) => `${k} - ${CRITERES_LABELS[k]} (${ponderations[k]}%)`)
      .join('\n   ');

    const wordCount = actualWordCount;
    const range = highlightRange;

    let message = `Corrige ce texte d'élève selon le protocole MEQ et attribue les cotes et la note finale automatiquement.\n\n`;
    message += `**Niveau:** ${niveau ? niveau.replace('sec', 'Secondaire ') : 'Non précisé'}\n`;
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

    message += `\n---\n**TEXTE DE L'ÉLÈVE À CORRIGER:**\n\n${texteEleve}`;

    onSubmit(message);
    onClose();
  };

  const canSubmit = texteEleve.trim().length > 0;

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
          {/* Niveau scolaire */}
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

          {/* Texte de l'élève */}
          <div className="space-y-1.5">
            <Label className="font-semibold text-gray-700">
              Texte de l'élève <span className="text-red-500">*</span>
            </Label>
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
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-2">
            <Button variant="outline" onClick={onClose} data-testid="correction-cancel-btn">
              Annuler
            </Button>
            <Button
              data-testid="correction-submit-btn"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-medium px-6"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1.5"><path d="M12 20h9"/><path d="M16.376 3.622a1 1 0 0 1 3.002 3.002L7.368 18.635a2 2 0 0 1-.855.506l-2.872.838a.5.5 0 0 1-.62-.62l.838-2.872a2 2 0 0 1 .506-.854z"/></svg>
              Corriger le texte
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TextCorrectionModal;
