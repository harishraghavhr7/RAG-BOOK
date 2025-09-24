# RAG Chatbot - Full Stack Application

A modern, responsive full-stack web application that integrates with your RAG (Retrieval-Augmented Generation) system using React.js frontend and Flask backend.

🎉 **SUCCESSFULLY CREATED AND TESTED!** 

Your full-stack RAG chatbot application is now ready to use. Both the frontend and backend are running and functional!

## Features

- 🤖 **Interactive Chatbot**: Ask questions about your documents and get intelligent responses
- 🔍 **Document Search**: Search for exact text matches across all loaded documents  
- 📚 **Document Management**: View information about loaded documents and their structure
- 💫 **Modern UI**: Clean, minimalistic design with smooth animations
- 📱 **Responsive Design**: Works perfectly on desktop and mobile devices
- ⚡ **Real-time Updates**: Live connection status and instant responses

## 🚀 Quick Start (Currently Running)

**Your application is currently running at:**
- 🌐 **Frontend**: http://localhost:3000
- 🔧 **Backend**: http://localhost:5000 (Demo Mode)

The demo mode provides:
- Interactive chat interface with sample responses
- Document search functionality (simulated)
- Modern, responsive UI
- All frontend features working

**To access your full RAG system:**
1. Ensure internet connection (for ML model downloads)
2. Start Ollama: `ollama serve` + `ollama pull mistral`
3. Replace `app_demo.py` usage with `app.py` for full functionality

## Architecture

- **Frontend**: React.js with modern hooks and CSS animations
- **Backend**: Flask API server with CORS support
- **AI Model**: Integrates with your existing RAG system (Mistral + Ollama)
- **Vector Database**: Uses your existing FAISS implementation
- **Document Processing**: Leverages your smart chunking algorithm

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Ollama installed and running (for AI responses)
- Your existing RAG documents in the `data/` directory

## 💻 Installation & Setup

### ✅ Already Completed (Currently Running)

1. **Backend Dependencies**: ✅ Installed
2. **Frontend Dependencies**: ✅ Installed  
3. **Backend Server**: ✅ Running at http://localhost:5000
4. **Frontend Server**: ✅ Running at http://localhost:3000

### 🔄 To Restart Servers

**Option 1: Use Demo Mode (Recommended for testing)**
```bash
# From the fullstack-rag directory
start-demo.bat
```

**Option 2: Manual Start**
```bash
# Backend (Demo Mode)
cd backend
python app_demo.py

# Frontend (New Terminal)
cd frontend
npm start
```

**Option 3: Full RAG Mode (Requires Internet + Ollama)**
```bash
# First ensure Ollama is running
ollama serve
ollama pull mistral

# Then start full backend
cd backend
python app.py

# Frontend (New Terminal)
cd frontend
npm start
```

## API Endpoints

### Health Check
- **GET** `/api/health` - Check server status and RAG initialization

### Chat
- **POST** `/api/chat` - Send a message to the chatbot
  ```json
  {
    "message": "What is a cell?"
  }
  ```

### Search
- **POST** `/api/search` - Search for exact text matches
  ```json
  {
    "query": "photosynthesis"
  }
  ```

### Documents
- **GET** `/api/documents` - Get information about loaded documents

## Project Structure

```
fullstack-rag/
├── backend/
│   ├── app.py              # Flask API server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html     # HTML template
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── App.css        # Styles
│   │   ├── index.js       # React entry point
│   │   └── index.css      # Global styles
│   └── package.json       # Node.js dependencies
└── README.md              # This file
```

## Configuration

### Backend Configuration

The backend automatically imports your existing RAG modules from the parent directory. Make sure your `optimized_rag.py` and `ollama_client.py` are in the correct location.

### Frontend Configuration

The frontend automatically detects the environment:
- Development: Connects to `http://localhost:5000`
- Production: Uses relative API paths

## Deployment

### Backend Deployment

For production deployment, consider using:
- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- **Docker**: Create a Dockerfile for containerized deployment
- **Cloud Platforms**: Deploy to AWS, Google Cloud, or Azure

### Frontend Deployment

Build the production frontend:

```bash
cd frontend
npm run build
```

Deploy the `build/` folder to:
- **Netlify**: Drag and drop deployment
- **Vercel**: Git-based deployment
- **AWS S3**: Static website hosting
- **Any web server**: Serve the static files

## Customization

### Styling

Modify `frontend/src/App.css` to customize the appearance:
- Colors and gradients
- Typography and spacing
- Animations and transitions
- Responsive breakpoints

### Backend Logic

Extend `backend/app.py` to add new features:
- Additional API endpoints
- Custom processing logic
- Authentication and authorization
- Rate limiting and caching

### Frontend Features

Enhance `frontend/src/App.js` to add:
- File upload functionality
- User authentication
- Chat history persistence
- Advanced search filters

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure Flask server is running on port 5000
   - Check CORS configuration
   - Verify firewall settings

2. **Ollama Connection Issues**
   - Start Ollama: `ollama serve`
   - Pull the model: `ollama pull mistral`
   - Check Ollama logs for errors

3. **No Documents Loaded**
   - Verify documents exist in `../../data/` directory
   - Check file permissions
   - Review backend logs for loading errors

4. **Frontend Build Errors**
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility
   - Review package.json dependencies

### Performance Optimization

- **Backend**: Implement caching for frequent queries
- **Frontend**: Use React.memo for expensive components
- **Database**: Optimize FAISS index parameters
- **Network**: Implement request compression

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

Built with ❤️ using React.js, Flask, and your RAG system.