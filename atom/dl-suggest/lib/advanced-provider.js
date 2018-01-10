'use babel';

// notice data is not being loaded from a local json file
// instead we will fetch suggestions from this URL
const API_URL = 'http://localhost:9999/v1/predict';

class AdvancedProvider {
	constructor() {
		// offer suggestions only when editing plain text or HTML files
		this.selector = '.text.plain, .source.python';

		// except when editing a comment within an HTML file
		//this.disableForSelector = '.comment';

		// make these suggestions appear above default suggestions
		this.suggestionPriority = 2;
	}

	getSuggestions(options) {
		const { editor, bufferPosition } = options;

		// getting the prefix on our own instead of using the one Atom provides
		let line = this.getCurrentLine(editor, bufferPosition);
		let shouldPredict = this.shouldPredict(line);

		// all of our snippets start with "tf"
		if (shouldPredict) {
			let sequence = this.getSequence(editor, bufferPosition);
			return this.findMatchingSuggestions(line, sequence);
		}
	}

	getCurrentLine(editor, bufferPosition) {
		let line = editor.getTextInRange([[bufferPosition.row, 0], bufferPosition]);
		let trimmedLine = line.trim();
		return trimmedLine;
	}

	shouldPredict(line) {
		// the prefix normally only includes characters back to the last word break
		// which is problematic if your suggestions include punctuation (like "@")
		// this expands the prefix back until a whitespace character is met
		// you can tweak this logic/regex to suit your needs
		if (line.startsWith('tf')) {
			if (line.search(/\(/) == -1) {
				return true;
			}
		}
		return false;
	}

	getSequence(editor, bufferPosition) {
		let startRow = Math.max(0, bufferPosition.row - 10);
		let sequence = editor.getTextInRange([[startRow, 0], bufferPosition]);
		return sequence;
	}

	findMatchingSuggestions(prefix, sequence) {
		// using a Promise lets you fetch and return suggestions asynchronously
		// this is useful for hitting an external API without causing Atom to freeze
		console.log(prefix);
		return new Promise((resolve) => {
			// fire off an async request to the external API
			let queryURL = API_URL + '?in=' + encodeURIComponent(sequence);
			fetch(queryURL)
				.then((response) => {
					// convert raw response data to json
					return response.json();
				})
				.then((json) => {
					// filter json (list of suggestions) to those matching the prefix
					var sugestions = [];
					let candidates = json['candidates'];
					for (candidate of candidates) {
						let code = candidate['code'];
						sugestions.push({
							text: code,
	        		replacementPrefix: prefix
						});
					}
					//console.log(sugestions);

					resolve(sugestions);

					//let matchingSuggestions = json.filter((suggestion) => {
					//	return suggestion.displayText.startsWith(prefix);
					//});

					// bind a version of inflateSuggestion() that always passes in prefix
					// then run each matching suggestion through the bound inflateSuggestion()
					//let inflateSuggestion = this.inflateSuggestion.bind(this, prefix);
					//let inflatedSuggestions = matchingSuggestions.map(inflateSuggestion);

					// resolve the promise to show suggestions
					//resolve(inflatedSuggestions);
				})
				.catch((err) => {
					// something went wrong
					console.log(err);
				});
		});
	}

	// clones a suggestion object to a new object with some shared additions
	// cloning also fixes an issue where selecting a suggestion won't insert it
	inflateSuggestion(replacementPrefix, suggestion) {
		return {
			displayText: suggestion.displayText,
			snippet: suggestion.snippet,
			description: suggestion.description,
			replacementPrefix: replacementPrefix, // ensures entire prefix is replaced
			iconHTML: '<i class="icon-comment"></i>',
			type: 'snippet',
			rightLabelHTML: '<span class="aab-right-label">Snippet</span>' // look in /styles/atom-slds.less
		};
	}

	onDidInsertSuggestion(options) {
		//atom.notifications.addSuccess(options.suggestion.displayText + ' was inserted.');
	}
}
export default new AdvancedProvider();
