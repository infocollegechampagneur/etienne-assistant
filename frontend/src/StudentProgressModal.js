import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Button } from './components/ui/button';

const ProgressChart = ({ profiles }) => {
  if (profiles.length < 2) return null;
  const W = 560, H = 160, PAD = 30;
  const maxPct = Math.max(...profiles.map(p => Math.max(p.pct_c4 || 0, p.pct_c5 || 0)), 5);
  const x = (i) => PAD + (i * (W - 2 * PAD)) / (profiles.length - 1);
  const y = (v) => H - PAD - ((v / maxPct) * (H - 2 * PAD));
  const line = (key) => profiles.map((p, i) => `${x(i)},${y(p[key] || 0)}`).join(' ');
  return (
    <div className="bg-gray-50 rounded-lg p-3" data-testid="progress-chart">
      <p className="text-xs font-semibold text-gray-600 mb-1">Évolution du % d'erreurs</p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#d1d5db" strokeWidth="1" />
        <text x={PAD - 5} y={PAD + 4} textAnchor="end" fontSize="9" fill="#9ca3af">{maxPct.toFixed(0)}%</text>
        <text x={PAD - 5} y={H - PAD} textAnchor="end" fontSize="9" fill="#9ca3af">0%</text>
        <polyline points={line('pct_c4')} fill="none" stroke="#f97316" strokeWidth="2" />
        <polyline points={line('pct_c5')} fill="none" stroke="#3b82f6" strokeWidth="2" />
        {profiles.map((p, i) => (
          <g key={i}>
            <circle cx={x(i)} cy={y(p.pct_c4 || 0)} r="3" fill="#f97316" />
            <circle cx={x(i)} cy={y(p.pct_c5 || 0)} r="3" fill="#3b82f6" />
            <text x={x(i)} y={H - PAD + 12} textAnchor="middle" fontSize="9" fill="#6b7280">#{i + 1}</text>
          </g>
        ))}
      </svg>
      <div className="flex gap-4 mt-1">
        <span className="text-[11px] text-gray-600 flex items-center gap-1"><span className="w-3 h-1 bg-orange-500 inline-block rounded"></span> C4 (Syntaxe + Ponctuation)</span>
        <span className="text-[11px] text-gray-600 flex items-center gap-1"><span className="w-3 h-1 bg-blue-500 inline-block rounded"></span> C5 (Usage + Grammaire)</span>
      </div>
    </div>
  );
};

const trendIcon = (profiles, key, i) => {
  if (i === 0) return null;
  const prev = profiles[i - 1][key] || 0;
  const cur = profiles[i][key] || 0;
  if (cur < prev) return <span className="text-green-600 ml-1">▼</span>;
  if (cur > prev) return <span className="text-red-500 ml-1">▲</span>;
  return <span className="text-gray-400 ml-1">＝</span>;
};

const StudentProgressModal = ({ open, onClose, apiUrl }) => {
  const [students, setStudents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [groupStats, setGroupStats] = useState(null);
  const [groupInput, setGroupInput] = useState('');
  const [goal, setGoal] = useState(null);
  const [goalC4, setGoalC4] = useState('');
  const [goalC5, setGoalC5] = useState('');
  const [goalSaved, setGoalSaved] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [importGroupe, setImportGroupe] = useState('');
  const [importNiveau, setImportNiveau] = useState('');
  const [importResult, setImportResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadStudents = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await axios.get(`${apiUrl}/student-profiles`);
      setStudents(res.data.students || []);
    } catch (e) {
      setError("Impossible de charger le suivi. Êtes-vous connecté?");
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    if (open) {
      setSelected(null);
      setProfiles([]);
      loadStudents();
    }
  }, [open, loadStudents]);

  const loadHistory = async (name) => {
    setSelected(name);
    setLoading(true);
    try {
      const res = await axios.get(`${apiUrl}/student-profiles/history`, { params: { name } });
      setProfiles(res.data.profiles || []);
      setGroupStats(res.data.group_stats || null);
      setGoal(res.data.goal || null);
      setGoalC4(res.data.goal?.target_pct_c4 ?? '');
      setGoalC5(res.data.goal?.target_pct_c5 ?? '');
      const g = (res.data.profiles || []).slice().reverse().find(p => p.groupe)?.groupe || '';
      setGroupInput(g);
    } catch (e) {
      setError("Erreur lors du chargement de l'historique");
    } finally {
      setLoading(false);
    }
  };

  const assignGroup = async () => {
    try {
      await axios.put(`${apiUrl}/student-profiles/assign-group`, { student_name: selected, groupe: groupInput.trim() });
      loadHistory(selected);
    } catch (e) {
      setError("Erreur lors de l'assignation du groupe");
    }
  };

  const saveGoal = async () => {
    try {
      await axios.put(`${apiUrl}/student-profiles/goal`, {
        student_name: selected,
        target_pct_c4: goalC4 === '' ? null : parseFloat(goalC4),
        target_pct_c5: goalC5 === '' ? null : parseFloat(goalC5)
      });
      setGoalSaved(true);
      setTimeout(() => setGoalSaved(false), 2000);
      loadHistory(selected);
    } catch (e) {
      setError("Erreur lors de l'enregistrement de l'objectif");
    }
  };

  const importClassFile = async (file) => {
    if (!file) return;
    setImportResult('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('groupe', importGroupe.trim());
      formData.append('niveau', importNiveau.trim());
      const res = await axios.post(`${apiUrl}/student-profiles/import-class`, formData);
      setImportResult(`✅ ${res.data.added} élève(s) ajouté(s)${res.data.skipped ? `, ${res.data.skipped} déjà existant(s)` : ''}`);
      loadStudents();
    } catch (e) {
      setImportResult(`❌ ${e.response?.data?.detail || "Erreur lors de l'import"}`);
    }
  };

  const deleteRosterStudent = async (rosterId) => {
    try {
      await axios.delete(`${apiUrl}/student-profiles/roster/${rosterId}`);
      loadStudents();
    } catch (e) {
      setError('Erreur lors de la suppression');
    }
  };

  const deleteEntry = async (profileId) => {
    try {
      await axios.delete(`${apiUrl}/student-profiles/entry/${profileId}`);
      const updated = profiles.filter(p => p.id !== profileId);
      setProfiles(updated);
      if (updated.length === 0) {
        setSelected(null);
        loadStudents();
      }
    } catch (e) {
      setError('Erreur lors de la suppression');
    }
  };

  const exportFile = async (format) => {
    try {
      const res = await axios.get(`${apiUrl}/student-profiles/export-${format === 'pdf' ? 'pdf' : 'excel'}`, {
        params: { name: selected },
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url;
      const safe = selected.replace(/[^\w-]/g, '_');
      a.download = format === 'pdf' ? `Suivi_${safe}.pdf` : `Profil_scripteur_${safe}.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setError(`Erreur lors de l'export ${format === 'pdf' ? 'PDF' : 'Excel'}`);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto" data-testid="student-progress-modal">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
            📈 Suivi des élèves — Profils scripteurs
          </DialogTitle>
          <p className="text-sm text-gray-500">Progression texte après texte. Les entrées s'ajoutent automatiquement quand vous corrigez avec l'option « Profil scripteur » et un nom d'élève.</p>
        </DialogHeader>

        {error && <p className="text-sm text-red-600 bg-red-50 rounded-lg p-2">{error}</p>}
        {loading && <p className="text-sm text-gray-500">Chargement...</p>}

        {!selected && !loading && (
          <div className="space-y-4">
            {/* Import de liste de classe */}
            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-500">
                {students.filter(s => s.alert).length > 0 && (
                  <span className="text-red-600 font-medium" data-testid="alerts-summary">
                    ⚠️ {students.filter(s => s.alert).length} élève(s) avec erreurs en hausse
                  </span>
                )}
              </p>
              <Button variant="outline" size="sm" onClick={() => setShowImport(!showImport)} data-testid="toggle-import-btn">
                📥 Importer une liste de classe
              </Button>
            </div>

            {showImport && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2" data-testid="import-panel">
                <p className="text-xs text-gray-600">Fichier Excel (.xlsx) ou CSV avec les noms des élèves dans la <strong>première colonne</strong>.</p>
                <div className="flex gap-2 items-center flex-wrap">
                  <input
                    type="text"
                    value={importGroupe}
                    onChange={(e) => setImportGroupe(e.target.value)}
                    placeholder="Groupe (ex: 32)"
                    className="w-28 h-8 text-xs border border-gray-300 rounded px-2"
                    data-testid="import-groupe-input"
                  />
                  <select
                    value={importNiveau}
                    onChange={(e) => setImportNiveau(e.target.value)}
                    className="h-8 text-xs border border-gray-300 rounded px-2 bg-white"
                    data-testid="import-niveau-select"
                  >
                    <option value="">Niveau (optionnel)</option>
                    <option value="sec1">Secondaire 1</option>
                    <option value="sec2">Secondaire 2</option>
                    <option value="sec3">Secondaire 3</option>
                    <option value="sec4">Secondaire 4</option>
                    <option value="sec5">Secondaire 5</option>
                  </select>
                  <input
                    type="file"
                    accept=".xlsx,.xls,.csv,.txt"
                    onChange={(e) => importClassFile(e.target.files?.[0])}
                    className="text-xs"
                    data-testid="import-file-input"
                  />
                </div>
                {importResult && <p className="text-xs font-medium" data-testid="import-result">{importResult}</p>}
              </div>
            )}

            {students.length === 0 ? (
              <div className="text-center py-8 text-gray-400 text-sm" data-testid="no-students-message">
                Aucun élève suivi pour le moment.<br />
                Corrigez un texte avec l'option <strong>Profil scripteur (PS)</strong> cochée et un <strong>nom d'élève</strong>, ou importez une liste de classe.
              </div>
            ) : (
              Object.entries(
                students.reduce((acc, s) => {
                  const key = s.groupe || 'Sans groupe';
                  (acc[key] = acc[key] || []).push(s);
                  return acc;
                }, {})
              ).sort(([a], [b]) => (a === 'Sans groupe') - (b === 'Sans groupe') || a.localeCompare(b)).map(([groupe, list]) => (
                <div key={groupe} className="space-y-2">
                  <p className="text-xs font-bold text-gray-500 uppercase tracking-wide flex items-center gap-2">
                    {groupe === 'Sans groupe' ? '📂 Sans groupe' : `🏫 Groupe ${groupe}`}
                    <span className="font-normal normal-case text-gray-400">({list.length} élève{list.length > 1 ? 's' : ''})</span>
                  </p>
                  {list.map(s => (
                    <div key={s.name} className="flex items-center gap-2">
                      <button
                        onClick={() => s.count > 0 && loadHistory(s.name)}
                        className={`flex-1 flex items-center justify-between p-3 rounded-lg border border-gray-200 transition-colors text-left ${s.count > 0 ? 'hover:border-orange-300 hover:bg-orange-50/40 cursor-pointer' : 'opacity-70 cursor-default'}`}
                        data-testid={`student-item-${s.name}`}
                      >
                        <div>
                          <p className="font-medium text-gray-800 flex items-center gap-2">
                            {s.name}
                            {s.alert && <span className="text-[10px] bg-red-100 text-red-600 rounded-full px-2 py-0.5 font-semibold" data-testid={`alert-badge-${s.name}`}>⚠️ Erreurs en hausse</span>}
                          </p>
                          <p className="text-xs text-gray-500">
                            {s.niveau ? s.niveau.replace('sec', 'Secondaire ') + ' · ' : ''}
                            {s.count > 0 ? `${s.count} texte${s.count > 1 ? 's' : ''} · dernier: ${s.last_date?.slice(0, 10)}` : 'Aucun texte corrigé encore'}
                          </p>
                        </div>
                        {s.count > 0 && <span className="text-orange-500">→</span>}
                      </button>
                      {s.roster_id && (
                        <button
                          onClick={() => deleteRosterStudent(s.roster_id)}
                          className="text-gray-300 hover:text-red-500 px-1"
                          title="Retirer cet élève de la liste"
                          data-testid={`delete-roster-${s.name}`}
                        >✕</button>
                      )}
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        )}

        {selected && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <button onClick={() => { setSelected(null); loadStudents(); }} className="text-sm text-gray-500 hover:text-gray-700" data-testid="back-to-students-btn">
                ← Tous les élèves
              </button>
              <div className="flex gap-2">
                <Button onClick={() => exportFile('pdf')} size="sm" className="bg-red-600 hover:bg-red-700 text-white" data-testid="export-pdf-btn">
                  🖨 Export PDF
                </Button>
                <Button onClick={() => exportFile('excel')} size="sm" className="bg-green-600 hover:bg-green-700 text-white" data-testid="export-excel-btn">
                  📥 Export Excel
                </Button>
              </div>
            </div>

            <div className="flex items-center justify-between gap-3">
              <h3 className="font-bold text-gray-800">{selected}</h3>
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500">Groupe:</span>
                <input
                  type="text"
                  value={groupInput}
                  onChange={(e) => setGroupInput(e.target.value)}
                  placeholder="Ex: 32"
                  className="w-20 h-7 text-xs border border-gray-300 rounded px-2"
                  data-testid="assign-group-input"
                />
                <button
                  onClick={assignGroup}
                  className="text-xs bg-orange-100 hover:bg-orange-200 text-orange-700 px-2 py-1 rounded transition-colors"
                  data-testid="assign-group-btn"
                >
                  Assigner
                </button>
              </div>
            </div>

            {goal?.achieved && profiles.length > 0 && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg p-3 text-center" data-testid="goal-celebration">
                <p className="text-sm font-bold text-green-700">🎉 Bravo! {selected} a atteint son objectif! 🎉</p>
                <p className="text-xs text-green-600 mt-0.5">Le dernier texte respecte les cibles fixées. Pensez à féliciter l'élève!</p>
              </div>
            )}

            {profiles.length >= 2 && (() => {
              const last = profiles[profiles.length - 1];
              const prev = profiles[profiles.length - 2];
              const up = ((last.pct_c4 || 0) + (last.pct_c5 || 0)) > ((prev.pct_c4 || 0) + (prev.pct_c5 || 0));
              return up ? (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2.5" data-testid="alert-banner">
                  <p className="text-xs font-medium text-red-700">⚠️ Attention : les erreurs de {selected} sont en hausse sur le dernier texte. Une intervention ciblée pourrait aider (voir les types d'erreurs fréquents ci-dessous).</p>
                </div>
              ) : null;
            })()}

            {/* Objectif personnalisé */}
            <div className="bg-amber-50/70 border border-amber-200 rounded-lg p-3" data-testid="goal-card">
              <p className="text-xs font-semibold text-amber-800 mb-2">🎯 Objectif personnalisé (% d'erreurs maximum visé)</p>
              <div className="flex items-center gap-2 flex-wrap">
                <label className="text-xs text-gray-600">C4 max:</label>
                <input
                  type="number" min="0" max="100" step="0.5"
                  value={goalC4}
                  onChange={(e) => setGoalC4(e.target.value)}
                  placeholder="%"
                  className="w-16 h-7 text-xs border border-gray-300 rounded px-2"
                  data-testid="goal-c4-input"
                />
                <label className="text-xs text-gray-600">C5 max:</label>
                <input
                  type="number" min="0" max="100" step="0.5"
                  value={goalC5}
                  onChange={(e) => setGoalC5(e.target.value)}
                  placeholder="%"
                  className="w-16 h-7 text-xs border border-gray-300 rounded px-2"
                  data-testid="goal-c5-input"
                />
                <button
                  onClick={saveGoal}
                  className="text-xs bg-amber-200 hover:bg-amber-300 text-amber-800 px-3 py-1 rounded transition-colors font-medium"
                  data-testid="save-goal-btn"
                >
                  {goalSaved ? '✅ Enregistré' : 'Fixer l\'objectif'}
                </button>
                {goal && !goal.achieved && profiles.length > 0 && (
                  <span className="text-[11px] text-amber-700">
                    Dernier texte: C4 {profiles[profiles.length - 1].pct_c4}%{goal.target_pct_c4 != null && ` / cible ${goal.target_pct_c4}%`} · C5 {profiles[profiles.length - 1].pct_c5}%{goal.target_pct_c5 != null && ` / cible ${goal.target_pct_c5}%`}
                  </span>
                )}
              </div>
            </div>

            {groupStats && profiles.length > 0 && (
              <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-3" data-testid="group-comparison-card">
                <p className="text-xs font-semibold text-indigo-800 mb-2">
                  🏫 Comparaison avec le groupe {groupStats.groupe} ({groupStats.students_count} élève{groupStats.students_count > 1 ? 's' : ''} suivi{groupStats.students_count > 1 ? 's' : ''})
                </p>
                <div className="grid grid-cols-2 gap-3">
                  {[['pct_c4', 'avg_pct_c4', 'C4 — Syntaxe + Ponctuation'], ['pct_c5', 'avg_pct_c5', 'C5 — Usage + Grammaire']].map(([k, avgK, label]) => {
                    const eleveVal = profiles[profiles.length - 1][k] || 0;
                    const avgVal = groupStats[avgK] || 0;
                    const better = eleveVal <= avgVal;
                    return (
                      <div key={k} className="bg-white rounded-lg p-2 border border-indigo-100">
                        <p className="text-[11px] text-gray-500">{label}</p>
                        <p className="text-sm font-bold text-gray-800">
                          {eleveVal}% <span className="text-[11px] font-normal text-gray-400">vs {avgVal}% (moy. groupe)</span>
                        </p>
                        <p className={`text-[11px] font-medium ${better ? 'text-green-600' : 'text-red-500'}`}>
                          {better ? '✓ Fait aussi bien ou mieux que le groupe' : '△ Plus d\'erreurs que la moyenne du groupe'}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <ProgressChart profiles={profiles} />

            <div className="overflow-x-auto rounded-lg border border-gray-200">
              <table className="w-full text-xs" data-testid="progress-table">
                <thead>
                  <tr className="bg-orange-500 text-white">
                    <th className="px-2 py-1.5 text-left">#</th>
                    <th className="px-2 py-1.5 text-left">Date</th>
                    <th className="px-2 py-1.5 text-left">Texte</th>
                    <th className="px-2 py-1.5 text-center">Mots</th>
                    <th className="px-2 py-1.5 text-center">S+P (C4)</th>
                    <th className="px-2 py-1.5 text-center">U+G (C5)</th>
                    <th className="px-2 py-1.5 text-center">% C4</th>
                    <th className="px-2 py-1.5 text-center">% C5</th>
                    <th className="px-2 py-1.5"></th>
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((p, i) => (
                    <tr key={p.id} className="even:bg-gray-50 border-t border-gray-100">
                      <td className="px-2 py-1.5 font-medium">{i + 1}</td>
                      <td className="px-2 py-1.5">{p.date?.slice(0, 10)}</td>
                      <td className="px-2 py-1.5 text-gray-500 max-w-[140px] truncate">{p.titre}</td>
                      <td className="px-2 py-1.5 text-center">{p.word_count}</td>
                      <td className="px-2 py-1.5 text-center">{p.total_sp}{trendIcon(profiles, 'total_sp', i)}</td>
                      <td className="px-2 py-1.5 text-center">{p.total_ug}{trendIcon(profiles, 'total_ug', i)}</td>
                      <td className="px-2 py-1.5 text-center">{p.pct_c4}%{trendIcon(profiles, 'pct_c4', i)}</td>
                      <td className="px-2 py-1.5 text-center">{p.pct_c5}%{trendIcon(profiles, 'pct_c5', i)}</td>
                      <td className="px-2 py-1.5 text-center">
                        <button onClick={() => deleteEntry(p.id)} className="text-gray-300 hover:text-red-500" title="Supprimer cette entrée" data-testid={`delete-entry-${p.id}`}>✕</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {profiles.length > 0 && (
              <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                <p className="text-xs font-semibold text-blue-800 mb-1">Types d'erreurs les plus fréquents</p>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(
                    profiles.reduce((acc, p) => {
                      Object.entries(p.error_counts || {}).forEach(([code, n]) => { acc[code] = (acc[code] || 0) + n; });
                      return acc;
                    }, {})
                  ).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([code, n]) => (
                    <span key={code} className="text-[11px] bg-white border border-blue-200 text-blue-700 rounded-full px-2 py-0.5">{code} × {n}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default StudentProgressModal;
