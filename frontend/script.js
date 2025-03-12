document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const inputText = document.getElementById('input-text');
    const loadingOverlay = document.getElementById('loading-overlay');

    analyzeBtn.addEventListener('click', async function() {
        const text = inputText.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        // Show loading state
        analyzeBtn.disabled = true;
        const btnText = analyzeBtn.querySelector('.btn-text');
        const loadingSpinner = analyzeBtn.querySelector('.loading-spinner');
        btnText.textContent = 'Analyzing...';
        loadingSpinner.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        try {
            // Make API call to your backend
            const response = await analyzeBias(text);
            updateUI(response);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the text');
        } finally {
            // Reset button state
            analyzeBtn.disabled = false;
            btnText.textContent = 'Analyze';
            loadingSpinner.classList.add('hidden');
            loadingOverlay.classList.add('hidden');
        }
    });

    async function analyzeBias(text) {
        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Response data:', data); // Debug log
            return data;
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }

    function updateUI(results) {
        try {
            console.log('Updating UI with results:', results); // Debug log

            // Update Bias Label
            const biasLabel = document.getElementById('bias-label');
            if (biasLabel) {
                biasLabel.textContent = results.bias_label || 'Not Available';
                biasLabel.className = 'value ' + (results.bias_label?.toLowerCase() === 'biased' ? 'biased' : 'unbiased');
            }

            // Update Bias Score - Modified to show exact score
            const biasScore = document.getElementById('bias-score');
            const biasScoreBar = document.getElementById('bias-score-bar');
            if (biasScore && biasScoreBar) {
                const score = parseFloat(results.bias_score);
                // Remove toFixed(2) to show exact score
                biasScore.textContent = isNaN(score) ? 'N/A' : score;
                if (!isNaN(score)) {
                    biasScoreBar.style.width = `${score * 100}%`;
                    biasScoreBar.style.backgroundColor = getScoreColor(score);
                }
            }

            // Update Bias Type
            const biasType = document.getElementById('bias-type');
            if (biasType) {
                biasType.textContent = results.bias_type || 'Not Specified';
            }

            // Update Explanation
            const biasExplanation = document.getElementById('bias-explanation');
            if (biasExplanation) {
                biasExplanation.textContent = results.explanation || 'No explanation available';
            }

            // Update Mitigation Strategies
            const mitigationStrategies = document.getElementById('mitigation-strategies');
            if (mitigationStrategies) {
                mitigationStrategies.innerHTML = formatMitigationStrategies(results.mitigation_strategies);
            }

            // Update RAG Results
            const ragResults = document.getElementById('rag-results');
            if (ragResults) {
                ragResults.innerHTML = `<p>${results.rag_results || 'No RAG results available'}</p>`;
            }

            // Update Web Results
            const webResults = document.getElementById('web-results');
            if (webResults) {
                webResults.innerHTML = `<p>${results.summary || 'No web results available'}</p>`;
            }

            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error updating UI:', error);
            alert('Error updating the results display');
        }
    }

    function formatMitigationStrategies(strategies) {
        if (!strategies) return '<p>No mitigation strategies available</p>';
        
        if (typeof strategies === 'string') {
            // Split the string by bullet points or newlines
            const strategyList = strategies.split(/[•\n]/).filter(s => s.trim());
            if (strategyList.length > 0) {
                return strategyList
                    .map(strategy => `<p>• ${strategy.trim()}</p>`)
                    .join('');
            }
            return `<p>${strategies}</p>`;
        }
        
        if (Array.isArray(strategies)) {
            return strategies
                .map(strategy => `<p>• ${strategy}</p>`)
                .join('');
        }
        
        return '<p>No mitigation strategies available</p>';
    }

    function getScoreColor(score) {
        // More granular color scale for precise scores
        if (score < 0.2) return '#00E676'; // Very low bias - Bright green
        if (score < 0.4) return '#4CAF50'; // Low bias - Green
        if (score < 0.6) return '#FFA726'; // Medium bias - Orange
        if (score < 0.8) return '#FF5722'; // High bias - Deep orange
        return '#F44336'; // Very high bias - Red
    }

    function formatAdditionalInfo(info) {
        if (!info) return '<p>No additional information available</p>';
        return `<p>${info}</p>`;
    }
});