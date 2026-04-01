import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './components/ui/dialog';
import { Button } from './components/ui/button';
import { Separator } from './components/ui/separator';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Label } from './components/ui/label';

const API = process.env.REACT_APP_BACKEND_URL + '/api';

const Nouveautes = ({ open, onClose, isAdmin }) => {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [editingNote, setEditingNote] = useState(null);
  const [form, setForm] = useState({ title: '', version: '', date: '', description: '', how_to_use: '' });

  useEffect(() => {
    if (open) fetchNotes();
  }, [open]);

  const fetchNotes = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/release-notes`);
      const data = await res.json();
      setNotes(data);
    } catch (err) {
      console.error('Erreur chargement nouveautés:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    const token = localStorage.getItem('etienne_token');
    const method = editingNote ? 'PUT' : 'POST';
    const url = editingNote ? `${API}/release-notes/${editingNote}` : `${API}/release-notes`;
    try {
      await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(form)
      });
      setEditMode(false);
      setEditingNote(null);
      setForm({ title: '', version: '', date: '', description: '', how_to_use: '' });
      fetchNotes();
    } catch (err) {
      console.error('Erreur sauvegarde:', err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Supprimer cette note ?')) return;
    const token = localStorage.getItem('etienne_token');
    try {
      await fetch(`${API}/release-notes/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      fetchNotes();
    } catch (err) {
      console.error('Erreur suppression:', err);
    }
  };

  const startEdit = (note) => {
    setForm({
      title: note.title,
      version: note.version,
      date: note.date,
      description: note.description,
      how_to_use: note.how_to_use || ''
    });
    setEditingNote(note.id);
    setEditMode(true);
  };

  const startAdd = () => {
    setForm({ title: '', version: '', date: new Date().toISOString().split('T')[0], description: '', how_to_use: '' });
    setEditingNote(null);
    setEditMode(true);
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto" data-testid="nouveautes-modal">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-orange-600"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
            Quoi de neuf ?
          </DialogTitle>
          <p className="text-sm text-gray-500">Découvrez les dernières fonctionnalités et améliorations d'Étienne</p>
        </DialogHeader>

        {/* Bouton ajouter (admin) */}
        {isAdmin && !editMode && (
          <Button onClick={startAdd} size="sm" className="bg-orange-600 hover:bg-orange-700 w-fit" data-testid="add-note-btn">
            + Ajouter une nouveauté
          </Button>
        )}

        {/* Formulaire d'édition */}
        {editMode && (
          <div className="bg-gray-50 rounded-lg p-4 space-y-3 border border-gray-200">
            <h3 className="font-semibold text-gray-800">{editingNote ? 'Modifier' : 'Ajouter'} une nouveauté</h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="col-span-1">
                <Label className="text-xs">Version</Label>
                <Input value={form.version} onChange={e => setForm({...form, version: e.target.value})} placeholder="Ex: 2.5" className="h-8" data-testid="note-version-input" />
              </div>
              <div className="col-span-1">
                <Label className="text-xs">Date</Label>
                <Input type="date" value={form.date} onChange={e => setForm({...form, date: e.target.value})} className="h-8" data-testid="note-date-input" />
              </div>
              <div className="col-span-1">
                <Label className="text-xs">Titre</Label>
                <Input value={form.title} onChange={e => setForm({...form, title: e.target.value})} placeholder="Titre court" className="h-8" data-testid="note-title-input" />
              </div>
            </div>
            <div>
              <Label className="text-xs">Description des changements</Label>
              <Textarea value={form.description} onChange={e => setForm({...form, description: e.target.value})} placeholder="Décrivez les nouveautés, une par ligne..." rows={4} className="text-sm" data-testid="note-description-input" />
            </div>
            <div>
              <Label className="text-xs">Comment utiliser (instructions)</Label>
              <Textarea value={form.how_to_use} onChange={e => setForm({...form, how_to_use: e.target.value})} placeholder="Expliquez comment utiliser les nouvelles fonctionnalités..." rows={3} className="text-sm" data-testid="note-howto-input" />
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" size="sm" onClick={() => { setEditMode(false); setEditingNote(null); }}>Annuler</Button>
              <Button size="sm" className="bg-orange-600 hover:bg-orange-700" onClick={handleSave} data-testid="save-note-btn">
                {editingNote ? 'Mettre à jour' : 'Publier'}
              </Button>
            </div>
          </div>
        )}

        <Separator />

        {/* Liste des notes */}
        {loading ? (
          <div className="text-center py-8 text-gray-400">Chargement...</div>
        ) : notes.length === 0 ? (
          <div className="text-center py-8 text-gray-400">Aucune nouveauté pour le moment.</div>
        ) : (
          <div className="space-y-6">
            {notes.map((note) => (
              <div key={note.id} className="relative" data-testid={`note-${note.id}`}>
                {/* Header */}
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center text-white font-bold text-sm">
                    {note.version || 'N'}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="font-bold text-gray-900">{note.title}</h3>
                      {note.version && (
                        <span className="text-xs bg-orange-100 text-orange-700 px-2 py-0.5 rounded-full font-medium">v{note.version}</span>
                      )}
                      <span className="text-xs text-gray-400">{note.date}</span>
                    </div>
                    
                    {/* Description */}
                    {note.description && (
                      <div className="mt-2 text-sm text-gray-700 space-y-1">
                        {note.description.split('\n').map((line, i) => {
                          const trimmed = line.trim();
                          if (!trimmed) return null;
                          if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
                            return <div key={i} className="flex gap-2 ml-1"><span className="text-orange-500 flex-shrink-0">&#x25B8;</span><span>{trimmed.replace(/^[-•]\s*/, '')}</span></div>;
                          }
                          if (trimmed.startsWith('## ') || trimmed.startsWith('**')) {
                            return <p key={i} className="font-semibold text-gray-900 mt-2">{trimmed.replace(/^##\s*|\*\*/g, '')}</p>;
                          }
                          return <p key={i}>{trimmed}</p>;
                        })}
                      </div>
                    )}

                    {/* Comment utiliser */}
                    {note.how_to_use && (
                      <div className="mt-3 bg-blue-50 border border-blue-100 rounded-lg p-3">
                        <p className="text-xs font-semibold text-blue-700 mb-1 flex items-center gap-1">
                          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                          Comment utiliser
                        </p>
                        <div className="text-sm text-blue-800 space-y-1">
                          {note.how_to_use.split('\n').map((line, i) => {
                            const trimmed = line.trim();
                            if (!trimmed) return null;
                            if (/^\d+\./.test(trimmed)) {
                              return <div key={i} className="flex gap-2 ml-1"><span className="font-semibold text-blue-600 flex-shrink-0">{trimmed.match(/^\d+\./)[0]}</span><span>{trimmed.replace(/^\d+\.\s*/, '')}</span></div>;
                            }
                            if (trimmed.startsWith('- ')) {
                              return <div key={i} className="flex gap-2 ml-3"><span className="text-blue-500 flex-shrink-0">&#x25B8;</span><span>{trimmed.replace(/^-\s*/, '')}</span></div>;
                            }
                            return <p key={i}>{trimmed}</p>;
                          })}
                        </div>
                      </div>
                    )}

                    {/* Actions admin */}
                    {isAdmin && (
                      <div className="flex gap-2 mt-2">
                        <button onClick={() => startEdit(note)} className="text-xs text-blue-600 hover:underline" data-testid={`edit-note-${note.id}`}>Modifier</button>
                        <button onClick={() => handleDelete(note.id)} className="text-xs text-red-500 hover:underline" data-testid={`delete-note-${note.id}`}>Supprimer</button>
                      </div>
                    )}
                  </div>
                </div>
                <Separator className="mt-4" />
              </div>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default Nouveautes;
