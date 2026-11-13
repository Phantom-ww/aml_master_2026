import { useState } from 'react';

function SaveInvestigationModal({ onSave, onClose }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [notes, setNotes] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!name.trim()) {
      alert('Please enter a name for this investigation');
      return;
    }
    onSave(name, description, notes);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
      <div className="bg-surface-2 rounded-xl border border-surface-4 w-full max-w-md shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-surface-3 flex items-center justify-between">
          <h2 className="text-lg font-bold text-white">Save Investigation</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Investigation Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Darknet cluster analysis"
              className="w-full px-4 py-2 bg-surface-3 border border-surface-4 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary"
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Description
            </label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description (optional)"
              className="w-full px-4 py-2 bg-surface-3 border border-surface-4 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Analyst Notes
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Compliance notes, case context, etc."
              rows={4}
              className="w-full px-4 py-2 bg-surface-3 border border-surface-4 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-accent-primary resize-none"
            />
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-surface-3 hover:bg-surface-4 text-white rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-accent-primary hover:bg-blue-500 text-white font-medium rounded-lg transition-colors"
            >
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default SaveInvestigationModal;
