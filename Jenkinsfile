pipeline {
    agent any
    environment {
        VENV_PATH = '/home/soumik/ml-devops-env/bin/activate'
    }
    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'github-token', url: 'https://github.com/Soumik1410/SPE_FInal_Project.git', branch: 'master'
            }
        }

        stage('Activate Venv') {
            steps {
                 sh '''
                    bash -c '
                    source ${VENV_PATH}
                    echo \$VIRTUAL_ENV
                    '
                '''
            }
        }
    }
}

