pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'github-token', url: 'https://github.com/Soumik1410/SPE_FInal_Project.git', branch: 'master'
            }
        }

        stage('Activate Venv') {
            steps {
                sh '''
                    source /home/soumik/ml-devops-env/bin/activate
                    echo $VIRTUAL_ENV
                '''
            }
        }
    }
}

