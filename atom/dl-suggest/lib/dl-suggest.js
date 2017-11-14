'use babel';

import DlSuggestView from './dl-suggest-view';
import { CompositeDisposable } from 'atom';

export default {

  dlSuggestView: null,
  modalPanel: null,
  subscriptions: null,

  activate(state) {
    this.dlSuggestView = new DlSuggestView(state.dlSuggestViewState);
    this.modalPanel = atom.workspace.addModalPanel({
      item: this.dlSuggestView.getElement(),
      visible: false
    });

    // Events subscribed to in atom's system can be easily cleaned up with a CompositeDisposable
    this.subscriptions = new CompositeDisposable();

    // Register command that toggles this view
    this.subscriptions.add(atom.commands.add('atom-workspace', {
      'dl-suggest:toggle': () => this.toggle()
    }));
  },

  deactivate() {
    this.modalPanel.destroy();
    this.subscriptions.dispose();
    this.dlSuggestView.destroy();
  },

  serialize() {
    return {
      dlSuggestViewState: this.dlSuggestView.serialize()
    };
  },

  toggle() {
    console.log('DlSuggest was toggled!');
    return (
      this.modalPanel.isVisible() ?
      this.modalPanel.hide() :
      this.modalPanel.show()
    );
  }

};
