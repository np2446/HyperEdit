const { exec } = require('child_process');
const path = require('path');

// Path to the CreateOperator.sh script
const scriptPath = path.join(__dirname, 'CreateOperator.sh');

// Path to Git Bash executable
const gitBashPath = 'C:\\Program Files\\Git\\bin\\bash.exe';

// Execute the shell script using Git Bash
exec(`"${gitBashPath}" ${scriptPath}`, (error, stdout, stderr) => {
    if (error) {
        console.error(`Error executing script: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`Script stderr: ${stderr}`);
        return;
    }
    console.log(`Script output: ${stdout}`);
});