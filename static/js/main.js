document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatbox = document.getElementById('chatbox');
    const typingIndicator = document.getElementById('typing-indicator');
    
    // Auto-scroll to bottom
    const scrollToBottom = () => {
        chatbox.scrollTop = chatbox.scrollHeight;
    };

    // Add message to DOM
    const appendMessage = (message, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<p>${escapeHTML(message)}</p>`;
        
        msgDiv.appendChild(contentDiv);
        
        // If it's a bot message, remove typing indicator first before adding new message
        if (sender === 'bot') {
            typingIndicator.classList.remove('active');
        }
        
        chatbox.appendChild(msgDiv);
        scrollToBottom();
    };

    // Prevent XSS
    const escapeHTML = (str) => {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    };

    // Form Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // User message
        appendMessage(message, 'user');
        userInput.value = '';
        
        // Show typing indicator
        chatbox.appendChild(typingIndicator);
        typingIndicator.classList.add('active');
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: 'local_user'
                })
            });

            const data = await response.json();
            
            // Artificial delay to simulate "thinking" for a better UX feeling
            setTimeout(() => {
                appendMessage(data.response || data.error || 'Server error occurred.', 'bot');
            }, 600);

        } catch (error) {
            console.error('Error:', error);
            setTimeout(() => {
                appendMessage('Sorry, I cannot connect to the backend right now.', 'bot');
            }, 600);
        }
    });
    
    // Initial scroll
    scrollToBottom();
});
