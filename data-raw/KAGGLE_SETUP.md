# Kaggle API Setup Guide

To download the Jigsaw Agile Community Rules dataset, you need to set up Kaggle API credentials.

## Steps

### 1. Create Kaggle Account
- Go to https://www.kaggle.com/
- Sign up or log in

### 2. Get API Token
- Go to https://www.kaggle.com/settings/account
- Scroll down to "API" section
- Click "Create New API Token"
- This will download `kaggle.json` file

### 3. Install Kaggle API Token

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Accept Competition Rules
- Go to https://www.kaggle.com/competitions/jigsaw-agile-community-rules/rules
- Click "I Understand and Accept" to accept the competition rules
- This is required before you can download the data

### 5. Run the Download Script

```bash
cd graniteR
Rscript data-raw/download_agile_rules.R
```

## Troubleshooting

### "Kaggle CLI not found"
Install Kaggle CLI:
```bash
pip install kaggle
# or
.venv/bin/pip install kaggle
```

### "401 - Unauthorized"
- Make sure kaggle.json is in ~/.kaggle/
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Verify JSON format is valid

### "403 - Forbidden"
- You haven't accepted the competition rules yet
- Visit the competition rules page and accept them
- Wait a few minutes and try again

### "Competition not found"
- Check the competition URL is correct
- Make sure you're logged in to Kaggle
- Competition might be archived or closed
