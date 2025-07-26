targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name which is used to generate a short unique hash for each resource')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
@metadata({
  azd: {
    type: 'location'
  }
})
param location string

param containerRegistryName string = ''
param aiHubName string = ''
@description('The Azure AI Studio project name. If ommited will be generated')
param aiProjectName string = ''
@description('The application insights resource name. If ommited will be generated')
param applicationInsightsName string = ''
@description('The Open AI resource name. If ommited will be generated')
param openAiName string = ''
@description('The Open AI connection name. If ommited will use a default value')
param openAiConnectionName string = ''
param keyVaultName string = ''
@description('The Azure Storage Account resource name. If ommited will be generated')
param storageAccountName string = ''

var abbrs = loadJsonContent('./abbreviations.json')
@description('The log analytics workspace name. If ommited will be generated')
param logAnalyticsWorkspaceName string = ''
param useApplicationInsights bool = true
param useContainerRegistry bool = true
var aiConfig = loadYamlContent('./ai.yaml')
@description('The name of the machine learning online endpoint. If ommited will be generated')
param resourceGroupName string = ''

@description('The API version of the OpenAI resource')
param openAiApiVersion string = '2023-07-01-preview'

@description('The name of the embedding model deployment')
param embeddingDeploymentName string = 'openai-text-embedding-ada-002'

@description('The name of the judge model deployment')
param judgeDeploymentName string = 'openai-gpt-4'

@description('The name of the teacher model deployment')
param teacherDeploymentName string = 'meta-llama-3-1-405B-chat'

@description('The name of the baseline model deployment')
param baselineDeploymentName string = 'meta-llama-2-7b-chat'

// List of models we know how to deploy
var allDeployments = array(contains(aiConfig, 'deployments') ? aiConfig.deployments : [])

// List of model names selected for deployment
var selectedDeploymentNames = [embeddingDeploymentName, judgeDeploymentName, teacherDeploymentName, baselineDeploymentName]

// Create role assignments for each deployment
var roleAssignments = {
  embedding: embeddingDeploymentName
  judge: judgeDeploymentName
  teacher: teacherDeploymentName
  baseline: baselineDeploymentName
}

// Group roles by deployment name using reduce
var deploymentRoleMapping = reduce(items(roleAssignments), {}, (acc, curr) => union(acc, {
  '${curr.value}': union(
    acc[?curr.value] ?? [],
    [curr.key]
  )
}))

// List of models selected for deployment
var filteredDeployments = filter(allDeployments, deployment => contains(selectedDeploymentNames, toLower(deployment.name)))

// Assign the specific roles that this deployment was selected for
var selectedDeployments = [for deployment in filteredDeployments: union(deployment, {
  roles: deploymentRoleMapping[deployment.name]
})]

@description('Id of the user or app to assign application roles')
param principalId string = ''

@description('Whether the deployment is running on GitHub Actions')
param runningOnGh string = ''

@description('Whether the deployment is running on Azure DevOps Pipeline')
param runningOnAdo string = ''

var resourceToken = take(toLower(uniqueString(subscription().id, environmentName, location)), 7)
var postfix = '${resourceToken}-${environmentName}'
var tags = { 'azd-env-name': environmentName }

resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// USER ROLES
var principalType = empty(runningOnGh) && empty(runningOnAdo) ? 'User' : 'ServicePrincipal'
module managedIdentity 'core/security/managed-identity.bicep' = {
  name: 'managed-identity'
  scope: resourceGroup
  params: {
    name: 'id-${postfix}'
    location: location
    tags: tags
  }
}

module ai 'core/host/ai-environment.bicep' = {
  name: 'ai'
  scope: resourceGroup
  params: {
    location: location
    tags: tags
    hubName: !empty(aiHubName) ? aiHubName : take('ai-hub-${postfix}', 32)
    projectName: !empty(aiProjectName) ? aiProjectName : take('ai-project-${postfix}', 32)
    keyVaultName: !empty(keyVaultName) ? keyVaultName : take('${abbrs.keyVaultVaults}${postfix}', 24)
    storageAccountName: !empty(storageAccountName)
      ? storageAccountName
      : take(replace('${abbrs.storageStorageAccounts}${postfix}', '-', ''), 24)
    openAiName: !empty(openAiName) ? openAiName : 'aoai-${postfix}'
    openAiConnectionName: !empty(openAiConnectionName) ? openAiConnectionName : 'aoai-connection'
    deployments: selectedDeployments
    logAnalyticsName: !useApplicationInsights
      ? ''
      : !empty(logAnalyticsWorkspaceName)
          ? logAnalyticsWorkspaceName
          : '${abbrs.operationalInsightsWorkspaces}${postfix}'
    applicationInsightsName: !useApplicationInsights
      ? ''
      : !empty(applicationInsightsName) ? applicationInsightsName : '${abbrs.insightsComponents}${postfix}'
    containerRegistryName: !useContainerRegistry
      ? ''
      : !empty(containerRegistryName) ? containerRegistryName : '${abbrs.containerRegistryRegistries}${postfix}'
    openaiApiVersion: openAiApiVersion
  }
}

module appinsightsAccountRole 'core/security/role.bicep' = {
  scope: resourceGroup
  name: 'appinsights-account-role'
  params: {
    principalId: managedIdentity.outputs.managedIdentityPrincipalId
    roleDefinitionId: '3913510d-42f4-4e42-8a64-420c390055eb' // Monitoring Metrics Publisher
    principalType: 'ServicePrincipal'
  }
}

module openaiRoleUser 'core/security/role.bicep' = if (!empty(principalId)) {
  scope: resourceGroup
  name: 'user-openai-user'
  params: {
    principalId: principalId
    roleDefinitionId: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd' //Cognitive Services OpenAI User
    principalType: principalType
  }
}

module openaiRoleContributor 'core/security/role.bicep' = if (!empty(principalId)) {
  scope: resourceGroup
  name: 'user-openai-contributor'
  params: {
    principalId: principalId
    roleDefinitionId: 'a001fd3d-188f-4b5d-821b-7da978bf7442' //Cognitive Services OpenAI Contributor
    principalType: principalType
  }
}

output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = resourceGroup.name

output AZURE_WORKSPACE_NAME string = ai.outputs.projectName

output APPINSIGHTS_CONNECTIONSTRING string = ai.outputs.applicationInsightsConnectionString

// Env variables are exported during postprocessing because bicep cannot conditionnally define outputs
// The names of the outputs for deployments depend on whether the platforn is openai or serverless
// For openai => AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_ENDPOINT
// For serverless => OPENAI_DEPLOYMENT and OPENAI_BASE_URL
output DEPLOYMENTS array = ai.outputs.deployments

// This is the Azure OpenAI endpoint used to fine tune student OpenAI models such as gpt-4o-mini
output FINETUNE_AZURE_OPENAI_ENDPOINT string = ai.outputs.openAiEndpoint
