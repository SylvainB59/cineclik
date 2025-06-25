const movieTitleInput = document.getElementById('movie_title_input');
const suggestionsList = document.getElementById('autocomplete-suggestions');
let timeoutId; // Pour le débounce


movieTitleInput.addEventListener('input', function () {
    clearTimeout(timeoutId); // Annule le timeout précédent
    const query = this.value.trim(); // Récupère le texte de l'input
    // console.log(query)

    if (query.length < 2) { // Commence à chercher à partir de 2 caractères
        suggestionsList.style.display = 'none'; // Cache les suggestions
        return;
    }

    // Débounce: attend 300ms après la dernière frappe
    timeoutId = setTimeout(() => {
        fetch(`/autocomplete?query=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(suggestions => {
                console.log(suggestions)
                suggestionsList.innerHTML = ''; // Vide les suggestions précédentes
                if (suggestions.length > 0) {
                    suggestions.forEach(title => {
                        const li = document.createElement('li');
                        li.textContent = title;
                        li.addEventListener('click', () => {
                            movieTitleInput.value = title; // Remplir l'input
                            suggestionsList.style.display = 'none'; // Cacher la liste
                            // Optionnel: soumettre le formulaire directement
                            // movieTitleInput.closest('form').submit(); 
                        });
                        suggestionsList.appendChild(li);
                    });
                    suggestionsList.style.display = 'block'; // Afficher la liste
                } else {
                    suggestionsList.style.display = 'none'; // Cacher si pas de résultats
                }
            })
            .catch(error => {
                console.error('Erreur lors de la récupération des suggestions:', error);
                suggestionsList.style.display = 'none';
            });
    }, 300); // Délai de 300 millisecondes
});

// Cacher la liste de suggestions quand l'input perd le focus
// Utiliser 'blur' est délicat avec les clics sur suggestions.
// Une meilleure approche est d'utiliser un événement de clic sur le document,
// et de vérifier si le clic est en dehors du conteneur d'autocomplétion.
document.addEventListener('click', function (event) {
    const autocompleteContainer = document.querySelector('.autocomplete-container');
    if (!autocompleteContainer.contains(event.target)) {
        suggestionsList.style.display = 'none';
    }
});

// S'assurer que les suggestions sont cachées si l'utilisateur navigue loin de l'input avec tabulation
movieTitleInput.addEventListener('focus', function () {
    // Afficher les suggestions seulement si l'input a déjà du texte
    if (this.value.trim().length >= 2 && suggestionsList.innerHTML !== '') {
        suggestionsList.style.display = 'block';
    }
});