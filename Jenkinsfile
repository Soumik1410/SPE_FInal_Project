pipeline {
    agent any
    stages {
        stage('Checkout Code') {
            steps {
                git credentialsId: 'github-token', url: 'https://github.com/Soumik1410/SPE_FInal_Project.git', branch: 'master'
            }
        }
        stage('Train Model') {
            steps {
                sh '/home/soumik/ml-devops-env/bin/python train.py'
            }
        }
        stage('Test Model') {
            steps {
                sh '/home/soumik/ml-devops-env/bin/python test.py'
            }
        }
    }
}

