pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'github-token', url: 'https://github.com/Soumik1410/SPE_FInal_Project.git', branch: 'master'
            }
        }

        stage('List Python Files') {
            steps {
                sh 'ls -l *.py'
            }
        }
    }
}

