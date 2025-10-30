// Additional AI Model Deployments
// This template deploys additional AI models to an existing Azure AI Services account

@description('The name of the existing Azure AI Services account')
param accountName string

@description('Array of additional model deployments to create')
param modelDeployments array

// Reference the existing AI Services account
resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: accountName
}

// Deploy additional models with sequential deployment to avoid conflicts
@batchSize(1)
resource additionalDeployments 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [for deployment in modelDeployments: {
  parent: account
  name: deployment.name
  properties: {
    model: deployment.model
  }
  sku: deployment.sku
}]

// Output the deployment names
output deploymentNames array = [for (deployment, i) in modelDeployments: additionalDeployments[i].name]
